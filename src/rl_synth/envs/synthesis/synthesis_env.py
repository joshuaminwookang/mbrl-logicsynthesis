import gym
import numpy as np
from gym import spaces
import os, glob, subprocess, json,sys, re
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET) # capture everything
logger.disabled = False

seq_list = [
    "&dc2", "&syn2", "&syn3", "&syn4", "&b", "&b -d",
    "&if -W 300 -x -K 6", "&if -W 300 -g -K 6", "&if -W 300 -y -K 6",
    "&synch2", "&dch", "&dch -f",
    "&if -W 300 -K 6 -v;&mfs;&st", "&if -W 300 -K 6 -v;&st", ""
]

pass_map = {
    '&dc2': 0, 
    '&syn2': 1, 
    '&syn3': 2, 
    '&syn4': 3, 
    '&b': 4, 
    '&b -d': 5, 
    '&if -W 300 -x -K 6': 6, 
    '&if -W 400 -g -K 6': 7, 
    '&if -W 400 -y -K 6': 8, 
    '&synch2': 9, 
    '&dch': 10, 
    '&dch -f': 11, 
    '&if -W 300 -K 6 -v;&mfs;&st': 12, 
    '&if -W 300 -K 6 -v;&st': 13, 
    '': 14
}

SEQ_TO_TOKEN = {
    '&if -W 300 -K 6 -v': 0, 
    '&st': 1, 
    '&synch2': 2, 
    '&dc2': 3, 
    '&if -W 300 -y -K 6': 4, 
    '&syn2': 5, 
    '&sweep': 6, 
    '&mfs': 7, 
    '&scorr': 8, 
    '&if -W 300 -g -K 6': 9, 
    '&b -d': 10, 
    '&if -W 300 -x -K 6': 11, 
    '&dch': 12, 
    '&b': 13, 
    '&syn4': 14, 
    '&dch -f': 15, 
    '&syn3': 16
}

TOKEN_TO_SEQ = {}
for x, y in SEQ_TO_TOKEN.items():
    TOKEN_TO_SEQ[y] = x

class SynthesisEnv(gym.Env):
    def __init__(self, exp_name, bmark):
        # Env Parameters
        self.env_name = 'synthesis'
        self.is_gym = True
        self.num_passes = 15
        self.num_features = 50 + self.num_passes
        self.action_dim = self.ac_dim = 1
        self.observation_dim = self.obs_dim = self.num_features #for hand-picked features; TODO change with Graph Embedding Dimension later
        self.eps = 0.1

        # Setup action and obs spaces
        self.action_space = gym.spaces.Discrete(self.num_passes)
        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.state = 0
        self.reward_dict = {}

        # Bookkeeping params
        self.run_dir = os.getcwd()
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        # self.bmark_path = glob.glob(self.script_dir +  "/verilog/*.v")[0]
        # self.bmark = os.path.basename(self.bmark_path).split('.')[0] # current circuit benchmark TODO: use Vivado
        self.bmark = bmark
        self.bmark_path = glob.glob(self.script_dir +  "/verilog/{}.v".format(bmark))[0]
        self.tag = exp_name
        if not os.path.exists(self.script_dir+'/temp'):
            os.makedirs(self.script_dir+'/temp')

        # save inital features of benchmark 
        self.baseline_rewards = 0.0
        self.baseline_rewards = -self.get_reward(np.zeros(self.obs_dim), np.array([-1]))[0][0]
        assert(1==0)
        self.baseline_obs = self._get_obs(self.state)
        self.last_obs = self.baseline_obs
        # print("Try new")
        # test_obs = np.tile(self.baseline_obs, (3, 1))
        # print(self.get_reward(test_obs, np.array([12,1,3]))[0] )
    def seed(self, seed):
        np.random.seed(seed)

    # action: np array of dimension (batchsize x 1)
    def step(self, action):
        # get reward of this action
        if isinstance(action, np.ndarray):
            ac = action[0]
        else:
            ac = action
        reward, done = self.get_reward(self.last_obs, action)
        # actual update: update state of env based on this action
        self.state = self.state *self.num_passes + ac + 1
        ob = self._get_obs(self.state)
        # print(ob)
        self.obs = ob
        # get score and write env_info
        score = self.get_score(ob)
        env_info = {'ob': ob,
                    'rewards': self.reward_dict,
                    'score': score}
        # print(str(reward) + "\n")
        return ob, reward, done, env_info

    def reset(self):
        self.state = 0
        self.last_obs = self.baseline_obs
        # self.counter = 0
        return self.last_obs

    # Function to run simulation and then get observations (separate from run sim + get reward as in get_reward())
    # action: np array of dimension (batchsize x 1)
    # @return: observations with dimension (ob_dim,) or (batch_size, ob_dim) 
    def run_sim_get_obs(self, actions):
        # Not batch mode: actions is a scalar value
        if not isinstance(actions, np.ndarray):
            state = self.state*self.num_passes + actions + 1
            if actions < self.num_passes: # unless stop token
                self._run_yosys(state)
            ob = self._get_obs(state)
            
        else:
            ob = []
            # Batch mode:  run Yosys for each action sequentially
            for ac in actions:
                ac = int(ac)
                state = self.state*self.num_passes + ac  +1
                ## run simulation and get rewards (Yosys)
                if ac < self.num_passes: # unless stop token
                    self._run_yosys(state)
                ob.append(self._get_obs(state))
            ob = np.asarray(ob)
        return ob

    # Run Yosys and then get reward from its log file
    # action: np array of dimension (batchsize x 1)
    # observations: np array 
    # state is always an integer (current state + seqeuence)
    def get_reward(self, observations, actions):
        delays = []
        areas = []
        totals  = []
        dones = []
        if not isinstance(actions, np.ndarray):
            state = self.state*self.num_passes + actions +1
            if actions < self.num_passes: # unless stop token
                self._run_yosys(state)
            try:
                fp = open("{}/temp/{}_{}.log".format(self.script_dir, self.bmark,self.tag), "r")
            except OSError:
                print("Could not open/read Yosys log")
                sys.exit()
            with fp:
                delay = float(re.findall(r'\d+.\d+', lines_that_contain("Del = ", fp)[-1])[0])
                fp.seek(0)
                area = float(re.findall(r'\d+.\d+', lines_that_contain("Ar = ", fp)[-1])[1])
                delays.append(delay)
                areas.append(area)
                #totals.append(-(delay*10000+area))
                totals.append(-(delay+area/10.0))
                dones.append(int(actions == self.num_passes-1 ))
        else:
            # Batch mode:  run Yosys for each action sequentially
            for ac in actions:
                ac = int(ac)
                # self.counter = self.counter + 1
                state = self.state*self.num_passes + ac +1
                # print("We are in get_reward, state is : " + str(state))
                ## run simulation and get rewards (Yosys)
                if ac < self.num_passes: # unless stop token
                    self._run_yosys(state)
                # if action is to terminate, no need to re-run yosys and get obervations
                # print("Iter : " + str(self.counter))
                # Now read yosys log to get Delay and Area
                try:
                    fp = open("{}/temp/{}_{}.log".format(self.script_dir, self.bmark,self.tag), "r")
                except OSError:
                    print("Could not open/read Yosys log")
                    sys.exit()
                with fp:
                    delay = float(re.findall(r'\d+.\d+', lines_that_contain("Del = ", fp)[-1])[0])
                    fp.seek(0)
                    area = float(re.findall(r'\d+.\d+', lines_that_contain("Ar = ", fp)[-1])[1])
                    delays.append(delay)
                    areas.append(area)
                    #totals.append(-(delay*10000+area))
                    totals.append(-(delay+area/10.0))
                    dones.append(int(ac == self.num_passes-1 ))

        self.reward_dict['r_delay'] = np.asarray(delays)
        self.reward_dict['r_area'] = np.asarray(areas)
        print( np.asarray(areas))
        print( np.asarray(delay))
        self.reward_dict['r_total'] = np.asarray(totals)
        dones = np.asarray(dones)
        return self.baseline_rewards - self.reward_dict['r_total'], dones

    def get_score(self, observations):
        # read abc results
        return observations[-1]

    ########################################
    # Env internal functions
    ########################################
    def _write_script(self, state):
        f = open(self.script_dir+"/temp/{}_{}.script".format(self.bmark,self.tag),"w")
        f.write(self._get_seq(state))
        f.close()
    
    def _get_histogram(self, state):
        i = state - 1
        histogram = np.zeros(self.num_passes)
        while i >= 0 :
            remainder = i % self.num_passes
            divisor = i // self.num_passes
            histogram[remainder] += 1
            if divisor <= 0 : 
                break;
            else : 
                i = divisor-1
        return histogram

    #@params: state (int)
    #@return: seq (str)
    def _get_seq(self, state):
        state = state - 1
        seq = ""
        while state >= 0 :
            remainder = state % self.num_passes
            divisor = state // self.num_passes
            seq = seq_list[remainder] + ";" + seq
            if divisor <= 0 : 
                break;
            else : 
                state = divisor-1
        seq = "&scorr;&sweep;"+ seq + "&if -K 6 -v;&mfs;\n&pfeatures {}/temp/stats_{}.json;&pfanstats {}/temp/fanstats_{}.json".format(self.script_dir, self.tag, self.script_dir,self.tag)
        return seq

    #@params: state (int)
    #@return: success or not
    def _run_abc(self, state):
        try:
            seq = self._get_seq(state)
            script = "read_blif {};&get;{}".format(self.bmark_path, seq)
            p = subprocess.check_output(["abc", "-c", script]) 
            # print(p)
            logger.info('ran')
            return True
        except:
            return False
 
    #@params: state (int)
    #@return: success or not           
    def _run_yosys(self, state):
        self._write_script(state)
        try:
            # p = subprocess.call(["yosys", "-p" ,"scratchpad -set abc9.script {}/{}.script; \
            #     synth_xilinx -dff -flatten -noiopad -abc9 -edif {}/{}.edif".format(self.script_dir, self.bmark, self.script_dir, self.bmark), \
            #         str(self.bmark_path), '-l', '{}/{}.log'.format(self.script_dir,self.bmark)], stdout=subprocess.DEVNULL) 
            p = subprocess.check_output(["yosys", "-p" ,"scratchpad -set abc9.script {}/temp/{}_{}.script; \
                synth_xilinx -dff -flatten -noiopad -abc9".format(self.script_dir, self.bmark, self.tag), \
                    str(self.bmark_path), '-l', '{}/temp/{}_{}.log'.format(self.script_dir,self.bmark,self.tag), "-q"]) 
            return True
        except:
            return False
    
    def _get_obs(self, state):
        # if (state == 0):
        #     return self.baseline_obs
        stats1 = glob.glob(os.path.normpath(self.script_dir + "/temp/stats_{}.json".format(self.tag)))
        stats2 = glob.glob(os.path.normpath(self.script_dir + "/temp/fanstats_{}.json".format(self.tag)))
        try:
            fp_stats1 =  open(stats1[0], "r")
            stats1_data = json.load(fp_stats1)
            fp_stats1.close()
        except:
            print("No stats.json for {} {}".format(self.bmark, state))
            return 0
        try:
            fp_stats2 =  open(stats2[0], "r")
            stats2_data = json.load(fp_stats2)
            fp_stats2.close()
        except:
            print("No fanstats.json for {} {}".format(self.bmark, state))
            return 0
        processed = preprocess_data({ **stats1_data, **stats2_data }) 
        # if (state > 0 ):
        #     return normalize(processed, self.baseline_obs)
        # return normalize(processed, processed)
        return np.concatenate((self._get_histogram(state), processed))

########################
# HELPER FUNCTIONS + PREPROCESSING
########################
def lines_that_contain(string, fp):
    return [line for line in fp if string in line]

# from pickle
def prepare_dataset(data):
    labels = []
    features = []
    sequences = []
    for _, d in data.items():
        feature = {}
        label = {}
        label['Path_Delay'] = d['Path_Delay'] / 1e2
        label['Slice_LUTs'] = d['Slice_LUTs'] / 1e3
        for f in ['CI', 'CO', 'level', 'level_avg',
                  'cut', 'xor', 'xor_ratio', 'mux', 'mux_ratio', 'and', 'and_ratio',
                  'obj', 'power', 'LUT', 'fanin', 'fanout', 'mffc', 
                  #'fanin_max', 'fanin_avg', 
                  'fanout_max', 'fanout_avg', 'mffc_max', 'mffc_avg']:
            feature[f] = d[f]
        labels.append(label)
        features.append(feature)
        sequences.append(d['Sequence'])
    return features, labels, sequences

def preprocess_faninout(d):
    fi = d.pop('fanin')
    fo = d.pop('fanout')
    d['fanin_all'] = fi['2'] # assume all fanin's are 2
    
    fo_threshold = 10
    fo_all, fo_large = 0, 0
    
    for k, v in fo.items():
        if k not in [str(x) for x in list(range(fo_threshold))]:
            fo_large += v
        fo_all += v
    
    for i in range(fo_threshold):
        d['fanout_ratio_%d' % i] = fo['%d' % i] / fo_all
    d['fanout_ratio_large'] = fo_large / fo_all
    d['fanout_all'] = fo_all
        
                
def preprocess_mffc(d):
    mffc = d.pop('mffc')
    mffc_threshold = 9
    mffc_all, mffc_large = 0, 0
    
    for k, v in mffc.items():
        if k not in [str(x) for x in list(range(mffc_threshold))]:
            mffc_large += v
        mffc_all += v
    
    for i in range(mffc_threshold):
        d['mffc_ratio_%d' % i] = 0. if mffc_all == 0 else mffc['%d' % i] / mffc_all
    d['mffc_ratio_large'] = 0. if mffc_all == 0 else mffc_large / mffc_all
    d['mffc_all'] = mffc_all
        
def preprocess_LUT(d):
    lut = d.pop('LUT')
    for i in [2, 3, 4, 5, 6]:
        key = f"{i}_LUT_ratio"
        lut_ratio = lut[key] if key in lut else 0.
        d[f"LUT_ratio_{i}"] = lut_ratio / 100 # convert percentage
    d["LUT_level"] = lut["level"]
    d["LUT_level_avg"] = lut["level_avg"]
    d["LUT_size_avg"] = lut["size_avg"]
    d["LUT_total"] = lut["total"]

def preprocess_sequence(seq):
    # convert the string representation into a list of tokens
    seq_list = []
    seq = seq.split(';')[2: -3] # remove the redundant parts
    sl = []
    for s in seq:
        if s.startswith('&'):
            sl.append(SEQ_TO_TOKEN[s])
    seq_list.append(np.array(sl))
    return seq_list

def flatten_all(d):
    # flattened_data = []
    # fd = list(d.values())
    # flattened_data.append(fd)
    # return np.array(flattened_data)
    return np.array(list(d.values()))

# def normalize(data):
#     eps = 1e-5
#     data_t = np.transpose(data)
#     for i in range(len(data_t)):
#         mean = np.mean(data_t[i])
#         std = np.std(data_t[i])
#         data_t[i] = (data_t[i] - mean) / (std + eps)
#     return np.transpose(data_t)

def normalize(data, baseline):
    eps = 1e-5
    return data - baseline

# features = dict of features
def preprocess_data(data):
    # with open(data_path, 'rb') as f:
    #     data = pickle.load(f)

    # features, labels, sequences = prepare_dataset(data)
    features = {}
    for f in ['CI', 'CO', 'level_avg', 'level', 
                  'cut', 'xor', 'xor_ratio', 'mux', 'mux_ratio', 'and', 'and_ratio',
                  'obj', 'power', 'LUT', 'fanin', 'fanout', 'mffc', 
                  #'fanin_max', 'fanin_avg', 
                  'fanout_max', 'fanout_avg', 'mffc_max', 'mffc_avg']:
        features[f] = data[f]
    preprocess_mffc(features)
    preprocess_faninout(features)
    # preprocess_slack(features)
    preprocess_LUT(features)

    features_flatted = flatten_all(features)
    # features_normalized = normalize(features_flatted)

    # sequences_list = preprocess_sequence(sequences)
    # labels_flattened = flatten_all(labels)
    return features_flatted
    return features_normalized, sequences_list, labels_flattened