import numpy as np
import pandas as pd
import glob, os.path, os
import argparse, re
import pickle, json, sys

def lines_that_contain(string, fp):
    return [line for line in fp if string in line]

def load_data_from_dir(dirname):
    data = {}
    subdirs =  glob.glob(os.path.normpath(dirname + "/tab_yosys-abc9*"))
    print("Found {} directories".format(len(subdirs)))
    
    # Get data from Yosys and Vivado logs
    i = 0
    for subdir in subdirs:
        this_data = {}
        vivado_log = glob.glob(os.path.normpath(subdir + "/test_5000.log"))
        script = glob.glob(os.path.normpath(subdir + "/*.abc.script"))
        # stats1 = glob.glob(os.path.normpath(subdir + "/stats.json"))
        # stats2= glob.glob(os.path.normpath(subdir + "/fanstats.json"))
        index = os.path.basename(script[0]).split('.')[1]
        bmark = os.path.basename(script[0]).split('.')[0]
        this_data['Index'] = index
        this_data['Benchmark'] = bmark
        try:
            fp_script = open(script[0], "r")
        except OSError:
            print("Could not open/read file:", script[0])
            sys.exit()
        with fp_script:
            try:
                this_data['Sequence'] = lines_that_contain("&scorr", fp_script)[0]
            except:
                print("No sequence file for {} {}".format(bmark,index))
                continue
        try:
            fp = open(vivado_log[0], "r")
        except OSError:
            print("Could not open/read file:", vivado_log[0])
            sys.exit()
        with fp:
            try:
                this_data["Path_Delay"] = float(re.findall(r'\d+.\d+', lines_that_contain("Path Delay", fp)[0])[0])
                fp.seek(0)
                this_data["Logic_Delay"] = float(re.findall(r'\d+.\d+', lines_that_contain("Logic Delay", fp)[0])[0])
                fp.seek(0)
                this_data["Net_Delay"] = float(re.findall(r'\d+.\d+', lines_that_contain("Net Delay", fp)[0])[0])  
                fp.seek(0)
                percentage = re.findall(r'\d+%', lines_that_contain("Logic Delay", fp)[0])[0]
                this_data["Logic_Delay_Percentage"] = float(percentage[:-1])
                fp.seek(0)
                this_data["Slice_LUTs"] = int(re.findall(r'\d+', lines_that_contain("Slice LUTs", fp)[0])[0])
                fp.seek(0)
                this_data["LUT_as_Logic"] = int(re.findall(r'\d+', lines_that_contain("LUT as Logic", fp)[0])[0])
                fp.seek(0)
                this_data["LUT_as_Memory"] = int(re.findall(r'\d+', lines_that_contain("LUT as Memory", fp)[0])[0])
                fp.seek(0)
                this_data["Regs_FF"] = int(re.findall(r'\d+', lines_that_contain("Register as Flip Flop", fp)[0])[0])
                fp.seek(0)
                this_data["Regs_Latch"] = int(re.findall(r'\d+', lines_that_contain("Register as Latch", fp)[0])[0])
            except:
                print("No Vivado log for {} {}".format(bmark, index))
                continue
        # try:
        #     fp_stats1 =  open(stats1[0], "r")
        #     stats1_data = json.load(fp_stats1)
        #     fp_stats1.close()
        # except:
        #     print("No stats.json for {} {}".format(bmark, index))
        #     continue
        # try:
        #     fp_stats2 =  open(stats2[0], "r")
        #     stats2_data = json.load(fp_stats2)
        #     fp_stats2.close()
        # except:
        #     print("No fanstats.json for {} {}".format(bmark, index))
        #     continue
        # this_data = {**this_data, **stats1_data, **stats2_data}
        data[i] = this_data
        i += 1
    return data


def main():
    parser = argparse.ArgumentParser(
            description='Create ABC script for each permuted sequence of synthesis transformations')
    parser.add_argument('--i', type=str, help='Results Directory')
    args = parser.parse_args()
    dir = os.path.abspath(args.i)

    bmark = os.path.basename(dir)
    #ip = bmark[bmark.find('run') + 4 :]
    ip = "medBmarks"
    data = load_data_from_dir(dir)

    # coalesce from 3 dirs
    data_arith =  load_data_from_dir("/global/scratch/users/minwoo_kang/run_epfl_arithmetic")
    data_ctrl =  load_data_from_dir("/global/scratch/users/minwoo_kang/run_epfl_control")
    data_vtr =  load_data_from_dir("/global/scratch/users/minwoo_kang/run_vtr_select")
    
    # df["Benchmark"] = ip
    print("outputting to: medBmarks.pkl")
    # df = pd.DataFrame.from_dict(df_dict)
    # df.to_csv(ip+".out.csv", sep="\t",index=False)

    with open(ip+".pkl", "wb") as handle:
        pickle.dump(data_arith + data_ctrl + data_vtr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ip + ".json", "w") as f:
        json.dump(data_arith + data_ctrl + data_vtr, f, indent=4)
    # Generate scatterplots for random runs
    # dfs = load_data_from_dir("results/random*.csv")


if __name__ == "__main__":
    main()
