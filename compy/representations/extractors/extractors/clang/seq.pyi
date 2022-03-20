import compy.representations.extractors.extractors

class ExtractionInfo:
    def __init__(self, *args, **kwargs) -> None: ...
    def accept(self, arg0: compy.representations.extractors.extractors.Visitor) -> None: ...
    @property
    def functionInfos(self) -> Any: ...

class FunctionInfo:
    def __init__(self, *args, **kwargs) -> None: ...
    def accept(self, arg0: compy.representations.extractors.extractors.Visitor) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def tokenInfos(self) -> Any: ...

class TokenInfo:
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def kind(self) -> str: ...
    @property
    def name(self) -> str: ...
