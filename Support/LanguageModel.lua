require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'misc.LSTM'



local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)


  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  local dropout = utils.getopt(opt, 'dropout', 0)

  self.seq_length = utils.getopt(opt, 'seq_length')

  self.core = LSTM.lstm(self.input_encoding_size, self.vocab_size + 1, self.rnn_size, self.num_layers, dropout)
  self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)
  self:_createInitState(1) 
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  
  if not self.init_state then self.init_state = {} end 
  for h=1,self.num_layers*2 do
   
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() 
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
end


function layer:createClones()
 
  print('constructing clones inside the LanguageModel')
  self.clones = {self.core}
  self.lookup_tables = {self.lookup_table}
  for t=2,self.seq_length+2 do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
  end
end

function layer:getModulesList()
  return {self.core, self.lookup_table}
end

function layer:parameters()
  
  local p1,g1 = self.core:parameters()
  local p2,g2 = self.lookup_table:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  
  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end

 

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end 
  for k,v in pairs(self.clones) do v:training() end
  for k,v in pairs(self.lookup_tables) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end 
  for k,v in pairs(self.clones) do v:evaluate() end
  for k,v in pairs(self.lookup_tables) do v:evaluate() end
end


function layer:sample(imgs, opt)
  local sample_max = utils.getopt(opt, 'sample_max', 1)
  local beam_size = utils.getopt(opt, 'beam_size', 1)
  local temperature = utils.getopt(opt, 'temperature', 1.0)
  if sample_max == 1 and beam_size > 1 then return self:sample_beam(imgs, opt) end 

  local batch_size = imgs:size(1)
  self:_createInitState(batch_size)
  local state = self.init_state

  
  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  local logprobs
  for t=1,self.seq_length+2 do

    local xt, it, sampleLogprobs
    if t == 1 then
      
      xt = imgs
    elseif t == 2 then
      
      it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      xt = self.lookup_table:forward(it)
    else
     
      if sample_max == 1 then
       
        sampleLogprobs, it = torch.max(logprobs, 2)
        it = it:view(-1):long()
      else
        
        local prob_prev
        if temperature == 1.0 then
          prob_prev = torch.exp(logprobs) 
        else
         
          prob_prev = torch.exp(torch.div(logprobs, temperature))
        end
        it = torch.multinomial(prob_prev, 1)
        sampleLogprobs = logprobs:gather(2, it) 
        it = it:view(-1):long()
      end
      xt = self.lookup_table:forward(it)
    end

    if t >= 3 then 
      seq[t-2] = it 
      seqLogprobs[t-2] = sampleLogprobs:view(-1):float() 
    end

    local inputs = {xt,unpack(state)}
    local out = self.core:forward(inputs)
    logprobs = out[self.num_state+1] 
    state = {}
    for i=1,self.num_state do table.insert(state, out[i]) end
  end

 
  return seq, seqLogprobs
end

function layer:sample_beam(imgs, opt)
  local beam_size = utils.getopt(opt, 'beam_size', 10)
  local batch_size, feat_dim = imgs:size(1), imgs:size(2)
  local function compare(a,b) return a.p > b.p end 

  assert(beam_size <= self.vocab_size+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  
  for k=1,batch_size do

    
    self:_createInitState(beam_size)
    local state = self.init_state

    local beam_seq = torch.LongTensor(self.seq_length, beam_size):zero()
    local beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size):zero()
    local beam_logprobs_sum = torch.zeros(beam_size) 
    local logprobs
    local done_beams = {}
    for t=1,self.seq_length+2 do

      local xt, it, sampleLogprobs
      local new_state
      if t == 1 then
       
        local imgk = imgs[{ {k,k} }]:expand(beam_size, feat_dim) 
        xt = imgk
      elseif t == 2 then
        
        it = torch.LongTensor(beam_size):fill(self.vocab_size+1)
        xt = self.lookup_table:forward(it)
      else
       
        local logprobsf = logprobs:float() 
        ys,ix = torch.sort(logprobsf,2,true) 
        local candidates = {}
        local cols = math.min(beam_size,ys:size(2))
        local rows = beam_size
        if t == 3 then rows = 1 end 
        for c=1,cols do 
          for q=1,rows do 
            local local_logprob = ys[{ q,c }]
            local candidate_logprob = beam_logprobs_sum[q] + local_logprob
            table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
          end
        end
        table.sort(candidates, compare) 

        
        new_state = net_utils.clone_list(state)
        local beam_seq_prev, beam_seq_logprobs_prev
        if t > 3 then
          
          beam_seq_prev = beam_seq[{ {1,t-3}, {} }]:clone()
          beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-3}, {} }]:clone()
        end
        for vix=1,beam_size do
          local v = candidates[vix]
          
          if t > 3 then
            beam_seq[{ {1,t-3}, vix }] = beam_seq_prev[{ {}, v.q }]
            beam_seq_logprobs[{ {1,t-3}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
          end
          
          for state_ix = 1,#new_state do
            
            new_state[state_ix][vix] = state[state_ix][v.q]
          end
          
          beam_seq[{ t-2, vix }] = v.c 
          beam_seq_logprobs[{ t-2, vix }] = v.r 
          beam_logprobs_sum[vix] = v.p 

          if v.c == self.vocab_size+1 or t == self.seq_length+2 then
            
            table.insert(done_beams, {seq = beam_seq[{ {}, vix }]:clone(), 
                                      logps = beam_seq_logprobs[{ {}, vix }]:clone(),
                                      p = beam_logprobs_sum[vix]
                                     })
          end
        end
        
        
        it = beam_seq[t-2]
        xt = self.lookup_table:forward(it)
      end

      if new_state then state = new_state end 

      local inputs = {xt,unpack(state)}
      local out = self.core:forward(inputs)
      logprobs = out[self.num_state+1]
      state = {}
      for i=1,self.num_state do table.insert(state, out[i]) end
    end

    table.sort(done_beams, compare)
    seq[{ {}, k }] = done_beams[1].seq 
    seqLogprobs[{ {}, k }] = done_beams[1].logps
  end

 
  return seq, seqLogprobs
end






function layer:updateOutput(input)
  local imgs = input[1]
  local seq = input[2]
  if self.clones == nil then self:createClones() end 
  assert(seq:size(1) == self.seq_length)
  local batch_size = seq:size(2)
  self.output:resize(self.seq_length+2, batch_size, self.vocab_size+1)
  
  self:_createInitState(batch_size)

  self.state = {[0] = self.init_state}
  self.inputs = {}
  self.lookup_tables_inputs = {}
  self.tmax = 0 
  for t=1,self.seq_length+2 do

    local can_skip = false
    local xt
    if t == 1 then
     
      xt = imgs 
    elseif t == 2 then
      
      local it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      self.lookup_tables_inputs[t] = it
      xt = self.lookup_tables[t]:forward(it) 
    else
   
      local it = seq[t-2]:clone()
      if torch.sum(it) == 0 then

        can_skip = true 
      end
    
      it[torch.eq(it,0)] = 1

      if not can_skip then
        self.lookup_tables_inputs[t] = it
        xt = self.lookup_tables[t]:forward(it)
      end
    end

    if not can_skip then
     
      self.inputs[t] = {xt,unpack(self.state[t-1])}
      
      local out = self.clones[t]:forward(self.inputs[t])
      
      self.output[t] = out[self.num_state+1] 
      self.state[t] = {} 
      for i=1,self.num_state do table.insert(self.state[t], out[i]) end
      self.tmax = t
    end
  end

  return self.output
end


function layer:updateGradInput(input, gradOutput)
  local dimgs 

  local dstate = {[self.tmax] = self.init_state} 
  for t=self.tmax,1,-1 do
    
    local dout = {}
    for k=1,#dstate[t] do table.insert(dout, dstate[t][k]) end
    table.insert(dout, gradOutput[t])
    local dinputs = self.clones[t]:backward(self.inputs[t], dout)
    
    local dxt = dinputs[1] 
    dstate[t-1] = {} 
    for k=2,self.num_state+1 do table.insert(dstate[t-1], dinputs[k]) end
    
    
    if t == 1 then
      dimgs = dxt
    else
      local it = self.lookup_tables_inputs[t]
      self.lookup_tables[t]:backward(it, dxt) 
    end
  end

  
  self.gradInput = {dimgs, torch.Tensor()}
  return self.gradInput
end


local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end


function crit:updateOutput(input, seq)
  self.gradInput:resizeAs(input):zero() 
  local L,N,Mp1 = input:size(1), input:size(2), input:size(3)
  local D = seq:size(1)
  assert(D == L-2, 'input Tensor should be 2 larger in time')

  local loss = 0
  local n = 0
  for b=1,N do 
    local first_time = true
    for t=2,L do 
     
      local target_index
      if t-1 > D then 
        target_index = 0
      else
        target_index = seq[{t-1,b}] 
      end
     
      if target_index == 0 and first_time then
        target_index = Mp1
        first_time = false
      end

     
      if target_index ~= 0 then
       
        loss = loss - input[{ t,b,target_index }] 
        self.gradInput[{ t,b,target_index }] = -1
        n = n + 1
      end

    end
  end
  self.output = loss / n 
  self.gradInput:div(n)
  return self.output
end

function crit:updateGradInput(input, seq)
  return self.gradInput
end

