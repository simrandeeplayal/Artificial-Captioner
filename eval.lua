require 'torch'
require 'nn'
require 'nngraph'
require 'loadcaffe'
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.DataLoaderRaw'
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')
cmd:option('-model','','path to model to evaluate')
cmd:option('-s_batch', 1, 'if > 0 then overrule, otherwise load from chkpt.')
cmd:option('-no_img', 100, 'how many images to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-dump_images', 1, 'Dump images into vis/imgs folder for vis? (1=yes,0=no)')
cmd:option('-dump_json', 1, 'Dump json with prdtn into vis folder? (1=yes,0=no)')
cmd:option('-dump_path', 0, 'Write image paths along with prdtn into vis json? (1=yes,0=no)')
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" prdtn.')
cmd:option('-image_folder', '', 'If this is nonempty then will predict on the images in this folder path')
cmd:option('-image_root', '', 'In case the image paths have to be preprended with a root path to an image folder')
cmd:option('-input_h5','','path to the h5file containing the preprocessed dataset. empty = fetch from model chkpt.')
cmd:option('-input_json','','path to the json file containing additional info and vocab. empty = fetch from model chkpt.')
cmd:option('-split', 'test', 'if running on MSCOCO images, which split to use: val|test|train')
cmd:option('-coco_json', '', 'if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', 'evalscript', 'an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) 
end

assert(string.len(opt.model) > 0, 'must provide a model')
local chkpt = torch.load(opt.model)
if string.len(opt.input_h5) == 0 then opt.input_h5 = chkpt.opt.input_h5 end
if string.len(opt.input_json) == 0 then opt.input_json = chkpt.opt.input_json end
if opt.s_batch == 0 then opt.s_batch = chkpt.opt.s_batch end
local fetch = {'rnn_size', 'input_encoding_size', 'drop_prob_lm', 'cnn_proto', 'cnn_model', 'seq_per_img'}
for k,v in pairs(fetch) do 
  opt[v] = chkpt.opt[v]
end
local vocab = chkpt.vocab
local ldr
if string.len(opt.image_folder) == 0 then
  ldr = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}
else  
  ldr = DataLoaderRaw{folder_path = opt.image_folder, coco_json = opt.coco_json}
end
local protos = chkpt.protos
protos.expander = nn.FeatExpander(opt.seq_per_img)
protos.crit = nn.LanguageModelCriterion()
protos.lm:createClones()
if opt.gpuid >= 0 then for k,v in pairs(protos) do v:cuda() end end
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local no_img = utils.getopt(evalopt, 'no_img', true)

  protos.cnn:evaluate()
  protos.lm:evaluate()
  ldr:resetIterator(split) 
  local z = 0
  local s_loss = 0
  local evalloss = 0
  local prdtn = {}
  while true do

    local data = ldr:getBatch{s_batch = opt.s_batch, split = split, seq_per_img = opt.seq_per_img}
    data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0) 
    z = z + data.images:size(1)

    
    local fts = protos.cnn:forward(data.images)

    
    local loss = 0
    if data.labels then
      local ex_fts = protos.expander:forward(fts)
      local lgpr = protos.lm:forward{ex_fts, data.labels}
      loss = protos.crit:forward(lgpr, data.labels)
      s_loss = s_loss + loss
      evalloss = evalloss + 1
    end
    local smpl_opr = { sample_max = opt.sample_max, beam_size = opt.beam_size, temperature = opt.temperature }
    local seq = protos.lm:sample(fts, smpl_opr)
    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1,#sents do
      local into = {image_id = data.infos[k].id, caption = sents[k]}
      if opt.dump_path == 1 then
        into.file_name = data.infos[k].file_path
      end
      table.insert(prdtn, into)
      if opt.dump_images == 1 then
        local cmd = 'cp "' .. path.join(opt.image_root, data.infos[k].file_path) .. '" vis/imgs/img' .. #prdtn .. '.jpg' -- bit gross
        print(cmd)
        os.execute(cmd)
      end
      if verbose then
        print(string.format('image %s: %s', into.image_id, into.caption))
      end
    end
    local indx0 = data.bounds.it_pos_now
    local indx1 = math.min(data.bounds.it_max, no_img)
    if verbose then
      print(string.format('evaluating performance... %d/%d (%f)', indx0-1, indx1, loss))
    end

    if data.bounds.wrapped then break end
    if no_img >= 0 and z >= no_img then break end 
  end

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(prdtn, opt.id)
  end

  return s_loss/evalloss, prdtn, lang_stats
end

local loss, splt_prdtn, lang_stats = eval_split(opt.split, {no_img = opt.no_img})
print('loss: ', loss)
if lang_stats then
  print(lang_stats)
end

if opt.dump_json == 1 then
  utils.write_json('vis/vis.json', splt_prdtn)
end
