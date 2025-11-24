"""
Progress tracking utilities for optimization.
"""

import sys
import time
from contextlib import contextmanager


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, step: int = None):
        """Update progress to a specific step or increment by 1."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        self._display()
    
    def _display(self):
        """Display current progress."""
        if self.total_steps == 0:
            return
        
        percent = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time
        
        if self.current_step > 0:
            avg_time_per_step = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            estimated_remaining = avg_time_per_step * remaining_steps
            eta_str = f" | ETA: {estimated_remaining/60:.1f}min"
        else:
            eta_str = ""
        
        # Simple progress line
        bar_length = 30
        filled = int(bar_length * self.current_step / self.total_steps)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        sys.stdout.write(f"\r{self.description}: [{bar}] {percent:.0f}% ({self.current_step}/{self.total_steps}){eta_str}")
        sys.stdout.flush()
        
        if self.current_step >= self.total_steps:
            print()  # New line when complete


@contextmanager
def track_progress(total_steps: int, description: str = "Progress"):
    """Context manager for tracking progress.
    
    Usage:
        with track_progress(10, "Processing") as tracker:
            for i in range(10):
                do_work()
                tracker.update()
    """
    tracker = ProgressTracker(total_steps, description)
    try:
        yield tracker
    finally:
        # Ensure we're at 100% when done
        if tracker.current_step < tracker.total_steps:
            tracker.update(tracker.total_steps)

