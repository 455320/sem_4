import os
import concurrent.futures
import time
import threading

class AdvancedFolderGenerator:
    """
    Generates a directory tree with a specified depth and width using parallelism.
    """
    
    MAX_DEPTH = 5
    MAX_WIDTH = 5
    BASE_DIR_NAME = "DesshOutput"

    def __init__(self, depth: int, width: int, base_path: str = None, max_workers: int = 64):
        self.depth = depth
        self.width = width
        self.max_workers = max_workers
        self.pending_count = 0
        self.condition = threading.Condition()
        
        if base_path:
             self.root_path = os.path.join(base_path, self.BASE_DIR_NAME)
        else:
             self.root_path = os.path.join(os.getcwd(), self.BASE_DIR_NAME)

    def validate_inputs(self) -> bool:
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
        result = ""
        while n >= 0:
            result = chr(n % 26 + ord('A')) + result
            n = n // 26 - 1
        return result

    def _task_done_callback(self, future):
        """Callback to decrement pending count when a task finishes."""
        with self.condition:
            self.pending_count -= 1
            if self.pending_count == 0:
                self.condition.notify_all()
        
        # Check for exceptions
        if future.exception():
             print(f"Task failed with: {future.exception()}")

    def generate(self):
        if not self.validate_inputs():
            return

        print(f"Generating folder structure at: {self.root_path}")
        print(f"Depth: {self.depth}, Width: {self.width}, Workers: {self.max_workers}")
        
        start_time = time.time()

        # Ensure base directory exists
        os.makedirs(self.root_path, exist_ok=True)
        
        # We manually manage the executor to avoid "shutdown before children submitted" issues
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
        try:
            with self.condition:
                self.pending_count += 1 # Count the initial task
                
            future = executor.submit(self._create_recursive_threaded, self.root_path, 1, executor)
            future.add_done_callback(self._task_done_callback)
            
            # Wait until all tasks are done
            with self.condition:
                while self.pending_count > 0:
                    self.condition.wait()
                    
        finally:
            executor.shutdown(wait=True)
        
        end_time = time.time()
        print(f"Done. Time taken: {end_time - start_time:.4f} seconds.")

    def _create_recursive_threaded(self, current_path: str, current_depth: int, executor):
        if current_depth > self.depth:
            return

        for i in range(self.width):
            col_name = self._get_column_name(i)
            folder_name = f"{col_name}{current_depth}"
            new_path = os.path.join(current_path, folder_name)
            
            try:
                os.makedirs(new_path, exist_ok=True)
                
                # Increment count BEFORE submitting to avoid race where count hits 0 temporarily
                with self.condition:
                    self.pending_count += 1
                
                future = executor.submit(self._create_recursive_threaded, new_path, current_depth + 1, executor)
                future.add_done_callback(self._task_done_callback)
                
            except OSError as e:
                print(f"Error creating {new_path}: {e}")

if __name__ == "__main__":
    try:
        depth = int(input("Enter depth (Max 5): "))
        width = int(input("Enter width (Max 5): "))
        generator = AdvancedFolderGenerator(depth, width)
        generator.generate()
    except ValueError:
        print("Invalid input.")
