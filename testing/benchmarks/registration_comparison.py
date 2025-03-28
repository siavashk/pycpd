from pycpd import RigidRegistration, DeformableRegistration
from pycpd import utility
from pycpd.utility import plot_metrics_comparison, plot_registration_comparison, get_slicer_positions_txt
import numpy as np
import time



def compare_registration_methods(X, Y, methods=None):
    """
    Compare different registration methods.

    Parameters:
        X: Target point cloud
        Y: Source point cloud
        methods: Dict of registration methods to compare
                e.g., {'rigid': RigidRegistration, 'deformable': DeformableRegistration}
    """
    if methods is None:
        methods = {
            'rigid': RigidRegistration,
            'deformable': DeformableRegistration
        }

    results = {}
    for name, RegistrationClass in methods.items():
        start_time = time.time()
        reg = RegistrationClass(**{'X': X, 'Y': Y})
        TY, params = reg.register()
        end_time = time.time()
        execution_time = end_time - start_time
        results[name] = {
            'transformed_points': TY,
            'parameters': params,
            'metrics': utility.calculate_registration_metrics(X, TY),
            'time': execution_time
        }

    # Visualize registration metrics and results
    plot_metrics_comparison(results)
    plot_registration_comparison(X, Y, results)

    return results

if __name__ == '__main__':
    X = np.loadtxt('../../data/fish_target.txt')
    Y = np.loadtxt('../../data/fish_source.txt')
    compare_registration_methods(X, Y)
