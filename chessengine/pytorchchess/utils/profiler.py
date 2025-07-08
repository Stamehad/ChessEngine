import time
import functools
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Optional
import threading

class SimpleProfiler:
    """Simple profiler that aggregates timings flatly and prints hierarchically"""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.component_stats = defaultdict(lambda: {'total_time': 0, 'call_count': 0})
        self._local = threading.local()
        self._original_methods = {}
        # self.reset()

    # def reset(self):
    #     """Reset all timing data"""
    #     self.component_stats = defaultdict(lambda: {'total_time': 0, 'call_count': 0})
    #     self._local = threading.local()

    def reset(self):
        """Reset all timing data and re-apply profiling decorators"""    
        self.component_stats.clear()
        self._local = threading.local()
        # for cls, methods in self._original_methods.items():
        #     for method_name, original_method in methods.items():
        #         profiled_method = profile(f"TorchBoard.{method_name}")(original_method)
        #         setattr(cls, method_name, profiled_method)

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
        try:
            yield
        finally:
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            if not hasattr(self, 'component_stats'):
                self.component_stats = defaultdict(lambda: {'total_time': 0, 'call_count': 0})
            self.component_stats[component]['total_time'] += elapsed
            self.component_stats[component]['call_count'] += 1

    def get_stats(self) -> Dict:
        """Get timing statistics"""
        if not self.enabled:
            return {"profiling": "disabled"}
        stats = {}
        for component, data in self.component_stats.items():
            stats[component] = {
                'total_time': data['total_time'],
                'call_count': data['call_count'],
                'avg_time': data['total_time'] / data['call_count'] if data['call_count'] > 0 else 0,
            }
        return stats

    def print_summary(self):
        """Print summary organized hierarchically by naming convention"""
        stats = self.get_stats()
        if not stats:
            print("No timing data collected")
            return

        # STEP PROFILING SUMMARY
        step_total = sum(data['total_time'] for comp, data in stats.items() if comp.startswith("step."))
        if step_total > 0:
            print("\n" + "="*80)
            print(f"STEP PROFILING SUMMARY (step total: {step_total:.3f}s)")
            print("="*80)
            print(f"{'Component':<45} {'Total(s)':<10} {'Calls':<8} {'Percentage':<12}")
            print("-" * 80)
            for component, data in sorted(stats.items(), key=lambda x: -x[1]['total_time']):
                if component.startswith("step."):
                    pct = (data['total_time'] / step_total * 100) if step_total > 0 else 0
                    print(f"{component:<45} {data['total_time']:<10.3f} "
                          f"{data['call_count']:<8} "
                          f"{pct:<.1f}%")

        # TORCHBOARD PROFILING SUMMARY (only TorchBoard.{method})
        torchboard_methods = {k: v for k, v in stats.items() if k.startswith("TorchBoard.")}
        # torchboard_total = stats.get("step.torch_board", {}).get("total_time", 0)
        torchboard_total = (
            stats.get("step.torch_board", {}).get("total_time", 0)
            or stats.get("full_run", {}).get("total_time", 0)
        )
        if torchboard_total > 0:
            print("\n" + "="*80)
            print(f"TORCHBOARD PROFILING SUMMARY (TorchBoard total: {torchboard_total:.3f}s)")
            print("="*80)
            print(f"{'Component':<45} {'Total(s)':<10} {'Calls':<8} {'Percentage':<12}")
            print("-" * 80)
            for component, data in sorted(torchboard_methods.items(), key=lambda x: -x[1]['total_time']):
                pct = (data['total_time'] / torchboard_total * 100) if torchboard_total > 0 else 0
                print(f"{component:<45} {data['total_time']:<10.3f} "
                      f"{data['call_count']:<8} "
                      f"{pct:<.1f}%")

        # Level 2: total
        total_time = sum(data['total_time'] for data in stats.values())
        print("\n" + "="*80)
        print(f"PROFILING SUMMARY (total time: {total_time:.3f}s)")
        print("="*80)
        print(f"{'Component':<45} {'Total(s)':<10} {'Calls':<8} {'Percentage':<12}")
        print("-"*80)
        for component, data in sorted(stats.items(), key=lambda x: -x[1]['total_time']):
            pct = (data['total_time'] / total_time * 100) if total_time > 0 else 0
            print(f"{component:<45} {data['total_time']:<10.3f} "
                  f"{data['call_count']:<8} "
                  f"{pct:<.1f}%")

# Global profiler instance
profiler = SimpleProfiler()

def profile(component_name: str):
    """Simple decorator that profiles at method/function boundaries"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profiler.time_block(component_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# def auto_profile_class(cls, method_profiles):
#     """
#     Automatically apply profiling decorators to class methods

#     Args:
#         cls: The class to modify
#         method_profiles: Dict mapping method names to profile component names
#     """
#     for method_name, component_name in method_profiles.items():
#         if hasattr(cls, method_name):
#             original_method = getattr(cls, method_name)
#             if callable(original_method):
#                 # Apply the profile decorator
#                 profiled_method = profile(component_name)(original_method)
#                 setattr(cls, method_name, profiled_method)
#     return cls

def auto_profile_class(cls, method_profiles):
    if cls not in profiler._original_methods:
        profiler._original_methods[cls] = {}
    for method_name, component_name in method_profiles.items():
        if hasattr(cls, method_name):
            original_method = getattr(cls, method_name)
            # Skip if already wrapped
            if getattr(original_method, "_is_profiled", False):
                continue
            # Save original
            profiler._original_methods[cls][method_name] = original_method
            # Wrap
            profiled_method = profile(component_name)(original_method)
            profiled_method._is_profiled = True
            setattr(cls, method_name, profiled_method)
    return cls