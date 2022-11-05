import json
import os

def write_unique_subject_filepaths(src_dir, output_filename):

    obj = {}
    asd_count = 0
    td_count = 0

    for dirpath, _, filenames in os.walk(src_dir):
        if "/asd" not in dirpath and "/td" not in dirpath:
            continue
        
        count = 0
        for filename in filenames:
            if filename in obj:
                count -= 1 # don't count repeats
            obj[filename] = dirpath
            count += 1
        
        if "/asd" in dirpath:
            asd_count += count
        elif "/td" in dirpath:
            td_count += count

    filepaths = []
    for filename, dirpath in obj.items():
        filepaths.append(dirpath + "/" + filename)


    with open(output_filename, "w") as fp:
        fp.write(json.dumps(filepaths))

    print(f"found {len(filepaths)} (ASD = {asd_count}, TD = {td_count}) unique subjects in {src_dir}")



src_dir = "../data/ABIDE/timeseries_filt_global/"
output_filename = "../data/ABIDE/unique.json"
write_unique_subject_filepaths(src_dir=src_dir, output_filename=output_filename)

src_dir = "../data/generated_filt_global/pearson_corr_raw/"
output_filename = "../data/generated_filt_global/pearson_corr_raw/unique.json"
write_unique_subject_filepaths(src_dir=src_dir, output_filename=output_filename)

src_dir = "../data/lanciano_datasets_corr_thresh_80/"
output_filename = "../data/lanciano_datasets_corr_thresh_80/unique.json"
write_unique_subject_filepaths(src_dir=src_dir, output_filename=output_filename)