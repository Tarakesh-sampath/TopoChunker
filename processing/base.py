from abc import ABC, abstractmethod

class BaseTopologyProcessor(ABC):
    """Base class for all topology-aware processors."""
    
    @abstractmethod
    def process(self, file_path: str) -> str:
        """Process a document and return a Unified Markdown Topology string."""
        pass
