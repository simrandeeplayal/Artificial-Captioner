require 'hdf5'
local utils = require 'misc.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)

prnt('DataLoader loading json file: ', opt.json_file)
  self.info = utils.read_json(opt.json_file)
  self.indx_wrd = self.info.indx_wrd
  self.vocab_size = utils.count_keys(self.indx_wrd)
  prnt('vocab size is ' .. self.vocab_size)

prnt('DataLoader loading h5 file: ', opt.h5_file)
  self.h5_file = hdf5.open(opt.h5_file, 'r')

 local img_s = self.h5_file:read('/images'):dataspaceSize()
  assert(#img_s == 4, '/images should be a 4D tensor')
  assert(img_s[3] == img_s[4], 'width and height must match')
  self.no_img = img_s[1]
  self.no_chnls = img_s[2]
  self.img_smax = img_s[3]
  prnt(strng.format('read %d images of size %dx%dx%d', self.no_img, 
            self.no_chnls, self.img_smax, self.img_smax))

 local s_seq = self.h5_file:read('/labels'):dataspaceSize()
  self.len_seq = s_seq[2]
  prnt('max sequence length in data is ' .. self.len_seq)

 self.start_indx = self.h5_file:read('/start_indx'):all()
  self.end_indx = self.h5_file:read('/end_indx'):all()

 self.splt_indx = {}
  self.iterators = {}
  for i,img in pairs(self.info.images) do
    local split = img.split
    if not self.splt_indx[split] then

 self.splt_indx[split] = {}
      self.iterators[split] = 1
    end
    table.insert(self.splt_indx[split], i)
  end
  for k,v in pairs(self.splt_indx) do
    prnt(strng.format('assigned %d images to split %s', #v, k))
  end
end

function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader:getVocabSize()
  return self.vocab_size
end

function DataLoader:getVocab()
  return self.indx_wrd
end

function DataLoader:getSeqLength()
  return self.len_seq
end

      function DataLoader:getBatch(opt)
  local split = utils.getopt(opt, 'split')
local s_batch = utils.getopt(opt, 's_batch', 5)
local seq_simg = utils.getopt(opt, 'seq_simg', 5)

  local splt_indx = self.splt_indx[split]
  assert(splt_indx, 'split ' .. split .. ' not found.')
 local raw_imgb = torch.ByteTensor(s_batch, 3, 256, 256)
  local batch = torch.LongTensor(s_batch * seq_simg, self.len_seq)
  local indx_mx = #splt_indx
  local wrap = false
  local infos = {}
  for i=1,s_batch do


    local r = self.iterators[split]

 local nxtr = r + 1
if nxtr > indx_mx then nxtr = 1; wrap = true end
self.iterators[split] = nxtr
    indx = splt_indx[r]
    assert(indx ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. r)
local img = self.h5_file:read('/images'):partial({indx,indx},{1,self.no_chnls},
                            {1,self.img_smax},{1,self.img_smax})
    raw_imgb[i] = img
local indx1 = self.start_indx[indx]
    local indx2 = self.end_indx[indx]
    local cpn = indx2 - indx1 + 1
assert(cpn > 0, 'an image does not have any label. this can be handled but rght now isn\'t')
    local seq
    if cpn < seq_simg then
 seq = torch.LongTensor(seq_simg, self.len_seq)
      for q=1, seq_simg do
        local indxl = torch.random(indx1,indx2)
        seq[{ {q,q} }] = self.h5_file:read('/labels'):partial({indxl, indxl}, {1,self.len_seq})
      end
    else
local indxl = torch.random(indx1, indx2 - seq_simg + 1)

 seq = self.h5_file:read('/labels'):partial({indxl, indxl+seq_simg-1}, {1,self.len_seq})
    end
    local ls = (i-1)*seq_simg+1
    batch[{ {ls,ls+seq_simg-1} }] = seq
local st_inf = {}
    st_inf.id = self.info.images[indx].id
    st_inf.file_path = self.info.images[indx].file_path
    table.insert(infos, st_inf)
  end

  local data = {}
  data.images = raw_imgb
  data.labels = batch:transpose(1,2):contiguous()
data.bounds = {it_pos_now = self.iterators[split], it_max = #splt_indx, wrap = wrap}
  data.infos = infos
  return data
end


