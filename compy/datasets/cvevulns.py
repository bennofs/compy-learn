import os
import subprocess
import pandas as pd
from tqdm import tqdm

from compy.representations.common import RepresentationBuilder
from compy.datasets import dataset


class CVEVulnsDataset(dataset.Dataset):
    def __init__(self):
        super().__init__()

        uri = "https://static.five.name/datasets/cvevulns-v2-57274a775006ed58fbecd1426442944100b9975c810dd7f402d804adf311ec1a.zip"
        self.download_http_and_extract(uri)

        self.additional_include_dirs = []

    def preprocess(self, builder: RepresentationBuilder, visitor, start_idx=-1, limit=None):
        # Load data about samples
        metadata = pd.read_json(os.path.join(self.content_dir, "metadata.json"), orient="index")
        functions = pd.read_json(os.path.join(self.content_dir, 'functions.json'))

        extracted_funcs = []
        no_pp_file = 0
        for idx, file_meta in tqdm(metadata.iterrows(), total=len(metadata), desc="Source Code -> IR+"):
            # FIXME: compile fail because they use throw. How to enable exception support?
            if idx in {4138, 4139}:
                continue

            if limit is not None and len(extracted_funcs) >= limit:
                break

            if idx < start_idx:
                continue
            if not file_meta.pp_name:
                no_pp_file += 1
                continue
            file_path = os.path.join(self.content_dir, "sources_pp", file_meta.pp_name)
            file_data = subprocess.run(
                ['clang', '-w', '-E', '-target', 'x86_64-pc-linux-gnu', file_path] + ["-D" + define for define in file_meta['parsed']['defines']],
                stdout=subprocess.PIPE,
                check=True,
            ).stdout
        
            # for .h files, try to detect whether to parse them as C++
            filename = file_meta.filename
            if filename.endswith(".h") and (b'template <typename' in file_data or b'template <class' in file_data):
                filename = filename + "xx"
                
            extraction_info = builder.string_to_info(file_data, filename=filename)
            for f in extraction_info.functionInfos:
                if f.name not in functions['functions'][idx]:
                    continue
                extracted_funcs.append({
                    "vulnerable": functions["vulnerable"][idx],
                    "info": f,
                    "file_idx": idx,
                })
        
        samples = []
        num_class = {
            True: 0, 
            False: 0
        }
        for func in tqdm(extracted_funcs, desc="IR+ -> ML Representation"):
            name = func["info"].name
            file_idx = func["file_idx"]
            rep = builder.info_to_representation(func['info'], visitor)
            num_class[func['vulnerable']] += 1
            samples.append({
                "info": {
                    "file_idx": file_idx,
                    "name": name,
                    "cve": metadata.cve[file_idx],
                },
                "x": {"code_rep": rep, "aux_in": []},
                "y": 1 if func["vulnerable"] else 0,
            })

        print("dataset size: {} ({} vulnerable and {} non-vulnerable, {:02.1f} percent vulnerable) functions".format(len(samples), num_class[True], num_class[False], num_class[True]/len(samples) * 100))
        print("Number of unique tokens:", builder.num_tokens())
        builder.print_tokens()

        return {
            "samples": samples,
            "num_types": builder.num_tokens(),
        }
