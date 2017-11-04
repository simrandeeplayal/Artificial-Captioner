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

cmd:option('-batch_size', 1, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_images', 100, 'how many images to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-dump_images', 1, 'Dump images into vis/imgs folder for vis? (1=yes,0=no)')
cmd:option('-dump_json', 1, 'Dump json with predictions into vis folder? (1=yes,0=no)')
cmd:option('-dump_path', 0, 'Write image paths along with predictions into vis json? (1=yes,0=no)')
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
