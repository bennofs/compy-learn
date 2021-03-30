import numpy as np

from sklearn.model_selection import StratifiedKFold

from compy import datasets as D
from compy import models as M
from compy import representations as R
from compy.representations.extractors import ClangDriver, clang_binary_path

# Load dataset
dataset = D.CVEVulnsDataset()

# Explore combinations
combinations = [
    # CGO 20: AST+DF, CDFG
    (R.ASTGraphBuilder, R.ASTDataVisitor, M.GnnPytorchGeomModel),
    (R.LLVMGraphBuilder, R.LLVMCDFGVisitor, M.GnnPytorchGeomModel),
    # Arxiv 20: ProGraML
    #(R.LLVMGraphBuilder, R.LLVMProGraMLVisitor, M.GnnPytorchGeomModel),
    # PACT 17: DeepTune
    (R.SyntaxSeqBuilder, R.SyntaxTokenkindVariableVisitor, M.RnnTfModel),
    # Extra
    #(R.ASTGraphBuilder, R.ASTDataCFGVisitor, M.GnnPytorchGeomModel),
    #(R.LLVMGraphBuilder, R.LLVMCDFGCallVisitor, M.GnnPytorchGeomModel),
    #(R.LLVMGraphBuilder, R.LLVMCDFGPlusVisitor, M.GnnPytorchGeomModel),
]

for builder, visitor, model in combinations:
    print("Processing %s-%s-%s" % (builder.__name__, visitor.__name__, model.__name__))

    # Build representation
    clang_driver = ClangDriver(
        ClangDriver.ProgrammingLanguage.C,
        ClangDriver.OptimizationLevel.O3,
        [],
        ['-target', 'x86_64-pc-linux-gnu', '-w']
    )
    clang_driver.setCompilerBinary(clang_binary_path())

    data = dataset.preprocess(builder(clang_driver), visitor)

    # downsample negative samples for 50/50 split
    samples_negative = [sample for sample in data["samples"] if not sample["y"]]
    samples_positive = [sample for sample in data["samples"] if sample["y"]]
    assert len(samples_negative) > len(samples_positive)

    rng = np.random.default_rng(seed=0)
    samples_negative_downsampled = rng.choice(samples_negative, size=len(samples_positive), replace=False)
    balanced_samples = np.append(samples_negative_downsampled, samples_positive)
    print("total samples: {}", len(balanced_samples))

    # Train and test
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=204)
    split = kf.split(balanced_samples, [sample["y"] for sample in balanced_samples])
    for train_idx, test_idx in split:
        model = model(num_types=data["num_types"])
        train_summary = model.train(
            list(np.array(balanced_samples)[train_idx]),
            list(np.array(balanced_samples)[test_idx]),
        )
        print(train_summary)

        break
