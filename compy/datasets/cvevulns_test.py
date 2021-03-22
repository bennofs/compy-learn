import pytest

from compy.datasets import CVEVulnsDataset
from compy.representations import RepresentationBuilder, ASTGraphBuilder, ASTDataVisitor


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


class TestBuilder(RepresentationBuilder):
    def string_to_info(self, src, additional_include_dir):
        functionInfo = objectview({"name": "xyz"})
        return objectview({"functionInfos": [functionInfo]})

    def info_to_representation(self, info, visitor):
        return "Repr"


@pytest.fixture
def cvevulns_fixture():
    ds = CVEVulnsDataset()
    yield ds


def test_preprocess(cvevulns_fixture):
    builder = TestBuilder()
    cvevulns_fixture.preprocess(builder, None)

if __name__ == '__main__':
    import sys
    builder = ASTGraphBuilder()
    visitor = ASTDataVisitor
    ds = CVEVulnsDataset()
    ds.preprocess(builder, visitor, start_idx=int(sys.argv[1] if len(sys.argv) == 2 else -1))
