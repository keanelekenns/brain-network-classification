import json
import os

def write_unique_subject_filepaths(src_dir, output_filename):

    asd_files = {}
    asd_count = 0
    td_files = {}
    td_count = 0

    for dirpath, _, filenames in os.walk(src_dir):

        if "/asd" in dirpath:
            for filename in filenames:
                if filename in asd_files:
                    asd_count -= 1 # don't count repeats
                asd_files[filename] = dirpath
                asd_count += 1
        elif "/td" in dirpath:
            for filename in filenames:
                if filename in td_files:
                    td_count -= 1 # don't count repeats
                td_files[filename] = dirpath
                td_count += 1
        

    asd_filepaths = []
    for filename, dirpath in asd_files.items():
        asd_filepaths.append(dirpath + "/" + filename)

    td_filepaths = []
    for filename, dirpath in td_files.items():
        td_filepaths.append(dirpath + "/" + filename)

    filepaths = {
        "ASD" : asd_filepaths,
        "TD" : td_filepaths
    }

    with open(output_filename, "w") as fp:
        fp.write(json.dumps(filepaths))

    print(f"found {asd_count + td_count} (ASD = {asd_count}, TD = {td_count}) unique subjects in {src_dir}")



src_dir = "./data/ABIDE/timeseries_filt_global/"
output_filename = "./data/ABIDE/unique.json"
write_unique_subject_filepaths(src_dir=src_dir, output_filename=output_filename)

src_dir = "./data/generated_filt_global/pearson_corr_raw/"
output_filename = "./data/generated_filt_global/pearson_corr_raw/unique.json"
write_unique_subject_filepaths(src_dir=src_dir, output_filename=output_filename)

src_dir = "./data/lanciano_datasets_corr_thresh_80/"
output_filename = "./data/lanciano_datasets_corr_thresh_80/unique.json"
write_unique_subject_filepaths(src_dir=src_dir, output_filename=output_filename)