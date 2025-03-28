from ..tests.benchmarks.registration_comparison import compare_registration_methods


def main():
    # Load your data
    fish_target = np.loadtxt('data/fish_target.txt')
    # ... set up X and Y ...

    # Run comparison
    results = compare_registration_methods(X, Y)

    # Print results
    for method_name, result in results.items():
        print(f"\n{method_name} Registration Results:")
        metrics = result['metrics']
        print(f"RMSE: {metrics['rmse']:.6f}")
        # ... print other metrics ...

    # Visualize
    plot_registration_comparison(X, Y, results)


if __name__ == "__main__":
    main()