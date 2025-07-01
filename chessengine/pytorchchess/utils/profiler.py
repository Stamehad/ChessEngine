import time
import functools
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Optional, List
import threading

class HierarchicalProfiler:
    """Clean hierarchical profiler that avoids double-counting"""
    
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.reset()
        self._local = threading.local()
    
    def reset(self):
        """Reset all timing data"""
        self.call_data = []  # List of (component, start_time, end_time, parent_idx)
        self.component_stats = defaultdict(lambda: {'total_time': 0, 'call_count': 0, 'self_time': 0})
        
    @property
    def _call_stack(self):
        """Thread-local call stack"""
        if not hasattr(self._local, 'stack'):
            self._local.stack = []
        return self._local.stack
        
    def enable(self):
        self.enabled = True
        
    def disable(self):
        self.enabled = False
        
    @contextmanager
    def time_block(self, component: str):
        """Context manager for timing code blocks"""
        if not self.enabled:
            yield
            return
            
        start_time = time.perf_counter()
        parent_idx = self._call_stack[-1] if self._call_stack else None
        current_idx = len(self.call_data)
        
        # Push to stack
        self._call_stack.append(current_idx)
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            
            # Record the call
            self.call_data.append((component, start_time, end_time, parent_idx))
            
            # Update stats
            self.component_stats[component]['total_time'] += elapsed
            self.component_stats[component]['call_count'] += 1
            
            # Pop from stack
            self._call_stack.pop()
    
    def _compute_self_times(self):
        """Compute self time (excluding children) for each component"""
        # Reset self times
        for stats in self.component_stats.values():
            stats['self_time'] = stats['total_time']
        
        # Subtract child times from parents
        for idx, (component, start, end, parent_idx) in enumerate(self.call_data):
            if parent_idx is not None:
                parent_component = self.call_data[parent_idx][0]
                child_time = end - start
                self.component_stats[parent_component]['self_time'] -= child_time
    
    def get_stats(self, self_time_only=True) -> Dict:
        """Get timing statistics"""
        if not self.enabled:
            return {"profiling": "disabled"}
        
        self._compute_self_times()
        
        stats = {}
        for component, data in self.component_stats.items():
            time_key = 'self_time' if self_time_only else 'total_time'
            stats[component] = {
                'time': data[time_key],
                'total_time': data['total_time'],
                'self_time': data['self_time'],
                'call_count': data['call_count'],
                'avg_time': data[time_key] / data['call_count'] if data['call_count'] > 0 else 0,
            }
        return stats
    
    def print_summary(self, top_n=10, self_time_only=True):
        """Print clean timing summary without double-counting"""
        if not self.enabled:
            print("Profiling disabled")
            return
            
        stats = self.get_stats(self_time_only)
        if not stats:
            print("No timing data collected")
            return
            
        print("\n" + "="*70)
        print("BEAM SEARCH PROFILING SUMMARY")
        print("="*70)
        
        time_key = 'time'
        time_label = 'Self(s)' if self_time_only else 'Total(s)'
        
        # Sort by the chosen time metric
        sorted_components = sorted(
            stats.items(), 
            key=lambda x: x[1][time_key], 
            reverse=True
        )[:top_n]
        
        print(f"{'Component':<30} {time_label:<10} {'Avg(ms)':<10} {'Calls':<8} {'%':<8}")
        print("-" * 70)
        
        total_time = sum(data[time_key] for _, data in sorted_components)
        
        for component, data in sorted_components:
            pct = (data[time_key] / total_time * 100) if total_time > 0 else 0
            print(f"{component:<30} {data[time_key]:<10.3f} "
                  f"{data['avg_time']*1000:<10.1f} {data['call_count']:<8} "
                  f"{pct:<8.1f}")

# Global profiler instance
profiler = HierarchicalProfiler()

def profile(component_name: str):
    """Simple decorator that only profiles at method/function boundaries"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profiler.time_block(component_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def auto_profile_class(cls, method_profiles):
    """
    Automatically apply profiling decorators to class methods
    
    Args:
        cls: The class to modify
        method_profiles: Dict mapping method names to profile component names
    """
    for method_name, component_name in method_profiles.items():
        if hasattr(cls, method_name):
            original_method = getattr(cls, method_name)
            if callable(original_method):
                # Apply the profile decorator
                profiled_method = profile(component_name)(original_method)
                setattr(cls, method_name, profiled_method)
    return cls