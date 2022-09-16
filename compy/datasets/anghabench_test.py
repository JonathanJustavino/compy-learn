import pytest

from compy.datasets import AnghabenchGraphDataset
from compy.representations import RepresentationBuilder


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


class TestBuilder(RepresentationBuilder):
    def string_to_info(self, src):
        functionInfo = objectview({"name": "xyz"})
        return objectview({"functionInfos": [functionInfo]})

    def info_to_representation(self, info, visitor):
        return "Repr"


@pytest.fixture
def anghabench_fixture():
    ds = AnghabenchGraphDataset()
    yield ds


def test_preprocess(anghabench_fixture):
    builder = TestBuilder()
    anghabench_fixture.preprocess(builder, None)
