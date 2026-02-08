"""
Folder Generator Script
Purpose: Generates a nested folder structure based on specified depth and width.
Disclaimer: This script creates directories effectively. Use with caution.
"""

import os
import shutil

class FolderGenerator:
    """
    Generates a directory tree with a specified depth and width.
    """
    
    MAX_DEPTH = 5
    MAX_WIDTH = 5
    BASE_DIR_NAME = "FolderGeneratorOutput"

    def __init__(self, depth: int, width: int, base_path: str = None):
        """
        Initialize the FolderGenerator.
        
        Args:
            depth (int): How deep the folder structure goes.
            width (int): How many folders to create at each level.
            base_path (str): Operations root directory. Defaults to CWD.
        """
        self.depth = depth
        self.width = width
        
        if base_path:
             self.root_path = os.path.join(base_path, self.BASE_DIR_NAME)
        else:
             self.root_path = os.path.join(os.getcwd(), self.BASE_DIR_NAME)

    def validate_inputs(self) -> bool:
        """Validates depth and width against safety limits."""
        if not isinstance(self.depth, int) or not isinstance(self.width, int):
            print("Error: Depth and Width must be integers.")
            return False
            
        if self.depth <= 0 or self.width <= 0:
            print("Error: Depth and Width must be positive integers.")
            return False

        if self.depth > self.MAX_DEPTH or self.width > self.MAX_WIDTH:
            print(f"Safety Limit Exceeded: Max Depth={self.MAX_DEPTH}, Max Width={self.MAX_WIDTH}")
            return False
            
        return True

    def _get_column_name(self, n: int) -> str:
        """Converts 0-based index to Excel-style column name (A, B, ... Z, AA...)."""
        result = ""
        while n >= 0:
            result = chr(n % 26 + ord('A')) + result
            n = n // 26 - 1
        return result

    def generate(self):
        """Executes the folder generation process."""
        if not self.validate_inputs():
            return

        print(f"Generating folder structure at: {self.root_path}")
        print(f"Depth: {self.depth}, Width: {self.width}")

        # Ensure base directory exists
        os.makedirs(self.root_path, exist_ok=True)
        
        # Start recursion
        self._create_recursive(self.root_path, current_depth=1)
        print("Done.")

    def _create_recursive(self, current_path: str, current_depth: int):
        """Recursive helper to create folder tree."""
        if current_depth > self.depth:
            return

        for i in range(self.width):
            col_name = self._get_column_name(i)
            # Naming format: {Column}{Row/Depth} -> e.g., A1, B1
            folder_name = f"{col_name}{current_depth}"
            new_path = os.path.join(current_path, folder_name)
            
            try:
                os.makedirs(new_path, exist_ok=True)
                # Recurse for the next level
                self._create_recursive(new_path, current_depth + 1)
            except OSError as e:
                print(f"Error creating {new_path}: {e}")

if __name__ == "__main__":
    try:
        d_input = input("Enter the depth of the folder structure (Max 5): ")
        w_input = input("Enter the width of each level (Max 5): ")
        
        depth = int(d_input)
        width = int(w_input)
        
        generator = FolderGenerator(depth, width)
        generator.generate()
        
    except ValueError:
        print("Invalid input. Please enter valid integers.")