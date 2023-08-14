from typing import List
from .superspace import SuperSpace
from .mobile_superspace import MobileSuperSpaceOriginal, MobileSuperSpaceV1, MobileSuperSpaceV1Res, MobileSuperSpaceW
from .onnx_superspace import OnnxSuperSpace, OnnxSuperSpaceV1, OnnxSuperSpaceW

superspace_dict = {
    MobileSuperSpaceOriginal.NAME: MobileSuperSpaceOriginal,
    MobileSuperSpaceV1.NAME: MobileSuperSpaceV1,
    MobileSuperSpaceV1Res.NAME: MobileSuperSpaceV1Res,
    MobileSuperSpaceW.NAME: MobileSuperSpaceW,
    OnnxSuperSpace.NAME: OnnxSuperSpace,
    OnnxSuperSpaceV1.NAME: OnnxSuperSpaceV1,
    OnnxSuperSpaceW.NAME: OnnxSuperSpaceW,
}


def get_available_superspaces() -> List[str]:
    return list(superspace_dict.keys())


def get_superspace(name: str) -> SuperSpace:
    assert name in get_available_superspaces(), f'{name} not in avail superspaces: {get_available_superspaces()}'
    return superspace_dict[name]()