from pycpd import RigidRegistration, DeformableRegistration
from pycpd.utility import calculate_registration_metrics


def main():
    """
    Main function to run registration comparison benchmarks.
    Tests multiple registration methods with different data scenarios.
    """
    import numpy as np
    import time
    from pathlib import Path
    import json
    from datetime import datetime

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return super().default(obj)

    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create results directory
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        fish_target = np.loadtxt('data/fish_target.txt')
        Y = np.zeros((fish_target.shape[0], fish_target.shape[1] + 1))
        Y[:, :-1] = fish_target

        # Create multiple test scenarios
        test_scenarios = {
            'rigid_transform': {
                'description': 'Pure rigid transformation',
                'transform': lambda points: np.dot(points,
                                                   np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                                                             [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                                                             [0, 0, 1]])) + np.array([1.0, 2.0, -1.0])
            },
            'deformed': {
                'description': 'Non-rigid deformation',
                'transform': lambda points: points + 0.1 * np.sin(points * 2)
            },
            'noisy': {
                'description': 'Rigid transform with noise',
                'transform': lambda points: (np.dot(points,
                                                    np.array([[np.cos(np.pi / 6), -np.sin(np.pi / 6), 0],
                                                              [np.sin(np.pi / 6), np.cos(np.pi / 6), 0],
                                                              [0, 0, 1]])) + np.array([0.5, 1.0, -0.5])
                                             + np.random.normal(0, 0.02, points.shape))
            }
        }

        # Registration methods to test
        registration_methods = {
            'rigid': RigidRegistration,
            'deformable': DeformableRegistration,
        }

        # Store all results
        all_results = {}

        # Run tests for each scenario
        for scenario_name, scenario in test_scenarios.items():
            logger.info(f"\nTesting scenario: {scenario_name}")
            logger.info(scenario['description'])

            # Create transformed version of points
            X = scenario['transform'](Y)
            scenario_results = {}

            # Test each registration method
            for method_name, RegistrationClass in registration_methods.items():
                logger.info(f"Running {method_name} registration...")

                try:
                    start_time = time.time()
                    reg = RegistrationClass(**{'X': X, 'Y': Y})
                    result = reg.register()

                    if method_name == 'rigid':
                        TY, (s_reg, R_reg, t_reg) = result
                        parameters = {
                            'scale': s_reg,
                            'rotation': R_reg,
                            'translation': t_reg
                        }
                    else:  # deformable
                        TY = reg.TY  # Get transformed points directly from the object
                        G, W = reg.get_registration_parameters()
                        parameters = {
                            'G': G,
                            'W': W
                        }

                    end_time = time.time()

                    metrics = calculate_registration_metrics(X, TY)

                    scenario_results[method_name] = {
                        'metrics': {
                            'rmse': float(metrics['rmse']),
                            'mae': float(metrics['mae']),
                            'max_error': float(metrics['max_error']),
                            'rmse_per_axis': metrics['rmse_per_axis'].tolist()
                        },
                        'runtime': float(end_time - start_time),
                        'success': True,
                        'parameters': parameters
                    }

                    logger.info(f"\n{method_name} Registration Results:")
                    logger.info(f"Runtime: {end_time - start_time:.2f} seconds")
                    logger.info(f"RMSE: {metrics['rmse']:.6f}")
                    logger.info(f"MAE: {metrics['mae']:.6f}")
                    logger.info(f"Max Error: {metrics['max_error']:.6f}")
                    logger.info("RMSE per axis: "
                                f"X={metrics['rmse_per_axis'][0]:.6f}, "
                                f"Y={metrics['rmse_per_axis'][1]:.6f}, "
                                f"Z={metrics['rmse_per_axis'][2]:.6f}")

                except Exception as e:
                    logger.error(f"Error in {method_name} registration: {str(e)}")
                    scenario_results[method_name] = {
                        'success': False,
                        'error': str(e)
                    }

            all_results[scenario_name] = scenario_results

        # Save results to JSON
        results_file = results_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)

        logger.info(f"\nResults saved to {results_file}")

        # Generate summary statistics
        logger.info("\nSummary Statistics:")
        for scenario_name, scenario_results in all_results.items():
            logger.info(f"\n{scenario_name}:")
            successful_methods = {k: v for k, v in scenario_results.items()
                                  if v['success']}

            if successful_methods:
                best_method = min(successful_methods.items(),
                                  key=lambda x: x[1]['metrics']['rmse'])
                logger.info(f"Best method: {best_method[0]}")
                logger.info(f"Best RMSE: {best_method[1]['metrics']['rmse']:.6f}")
            else:
                logger.warning("No successful registrations")

    except Exception as e:
        logger.error(f"Error in benchmark: {str(e)}")
        raise


if __name__ == "__main__":
    main()