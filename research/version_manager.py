"""
Version Management System
=========================

Handles automatic versioning of plots, results, and project naming.
"""

import os
import re
from typing import Tuple, Optional


class VersionManager:
    """Manages versioning for plots, results, and projects"""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
    
    def get_next_version(self, prefix: str = "V") -> str:
        """Get the next version number for a given prefix"""
        if not os.path.exists(self.base_dir):
            return f"{prefix}0001"
        
        # Find existing versions by looking for files that contain the version pattern
        existing_versions = []
        for item in os.listdir(self.base_dir):
            # Look for pattern like "_V0001" in filenames
            import re
            pattern = rf"_{prefix}(\d{{4}})"
            match = re.search(pattern, item)
            if match:
                version_num = int(match.group(1))
                existing_versions.append(version_num)
        
        if not existing_versions:
            return f"{prefix}0001"
        
        # Get next version number
        next_version = max(existing_versions) + 1
        return f"{prefix}{next_version:04d}"
    
    def get_versioned_filename(self, base_name: str, extension: str = "", prefix: str = "V") -> str:
        """Get a versioned filename"""
        version = self.get_next_version(prefix)
        if extension:
            return f"{base_name}_{version}.{extension}"
        else:
            return f"{base_name}_{version}"
    
    def get_next_project_name(self, base_name: str = "mach") -> str:
        """Get the next project name (e.g., mach1, mach2, etc.)"""
        if not os.path.exists(self.base_dir):
            return f"{base_name}1"
        
        # Find existing project directories
        existing_projects = []
        for item in os.listdir(self.base_dir):
            if os.path.isdir(os.path.join(self.base_dir, item)):
                # Check if it matches the pattern (e.g., mach1, mach2, mach3_title, etc.)
                match = re.match(rf"^{base_name}(\d+)(?:_.*)?$", item)
                if match:
                    existing_projects.append(int(match.group(1)))
        
        if not existing_projects:
            return f"{base_name}1"
        
        # Get next project number
        next_number = max(existing_projects) + 1
        return f"{base_name}{next_number}"
    
    def create_versioned_directories(self, project_name: str) -> Tuple[str, str, str]:
        """Create versioned directories for a project"""
        project_dir = os.path.join(self.base_dir, project_name)
        results_dir = os.path.join(project_dir, "results")
        plots_dir = os.path.join(project_dir, "plots")
        
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        return project_dir, results_dir, plots_dir
