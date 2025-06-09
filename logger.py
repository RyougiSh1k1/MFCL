import os
import logging
import datetime
from pathlib import Path


def setup_logger(args):
    """
    Setup logger with file and console handlers
    
    Args:
        args: argument parser object containing experiment parameters
        
    Returns:
        logger: configured logger object
        log_dir: path to log directory
    """
    # Create output directory structure
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment name based on key parameters
    exp_name = f"{args.method}_{args.dataset}_clients{args.num_clients}_tasks{args.n_tasks}_seed{args.seed}_{timestamp}"
    
    # Create output directory
    output_dir = Path("outputs")
    log_dir = output_dir / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger('MFCL')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(message)s')
    
    # File handler for detailed logs
    log_file = log_dir / "experiment.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler for simple output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Log experiment configuration
    logger.info("="*80)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("="*80)
    
    # Log all arguments
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    
    logger.info("="*80)
    logger.info(f"Experiment directory: {log_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    return logger, log_dir


def log_task_results(logger, task_id, round_num, accuracy, method):
    """Log results for a specific task and round"""
    logger.info(f"Task {task_id} | Round {round_num} | Method: {method} | Accuracy: {accuracy:.2f}%")


def log_forgetting_results(logger, task_id, accuracies, avg_forgetting):
    """Log forgetting analysis results"""
    logger.info("-"*60)
    logger.info(f"TASK {task_id} COMPLETION - FORGETTING ANALYSIS")
    logger.info("-"*60)
    
    for i, acc in enumerate(accuracies):
        logger.info(f"Task {i} accuracy: {acc:.2f}%")
    
    logger.info(f"Average forgetting: {avg_forgetting:.2f}%")
    logger.info("-"*60)


def log_final_results(logger, max_accuracy, final_accuracies, n_tasks):
    """Log final experiment results"""
    logger.info("="*80)
    logger.info("FINAL EXPERIMENT RESULTS")
    logger.info("="*80)
    
    total_forgetting = sum([max_accuracy[i] - final_accuracies[i] for i in range(n_tasks)]) / n_tasks
    
    logger.info("Maximum accuracies achieved per task:")
    for i, max_acc in enumerate(max_accuracy):
        logger.info(f"Task {i}: {max_acc:.2f}%")
    
    logger.info("\nFinal accuracies per task:")
    for i, final_acc in enumerate(final_accuracies):
        logger.info(f"Task {i}: {final_acc:.2f}%")
    
    logger.info(f"\nOverall average forgetting: {total_forgetting:.2f}%")
    logger.info("="*80)


def save_experiment_config(args, log_dir):
    """Save experiment configuration to a file"""
    config_file = log_dir / "config.txt"
    
    with open(config_file, 'w') as f:
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("="*50 + "\n")
        
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")
        
        f.write("="*50 + "\n")


def save_results_summary(log_dir, max_accuracy, final_accuracies, n_tasks, method):
    """Save results summary to CSV and text files"""
    import csv
    
    # Save to CSV
    csv_file = log_dir / "results_summary.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'Max_Accuracy', 'Final_Accuracy', 'Forgetting'])
        
        for i in range(n_tasks):
            forgetting = max_accuracy[i] - final_accuracies[i] if i < len(final_accuracies) else 0
            writer.writerow([i, f"{max_accuracy[i]:.2f}", 
                           f"{final_accuracies[i]:.2f}" if i < len(final_accuracies) else "N/A", 
                           f"{forgetting:.2f}"])
        
        # Add average forgetting
        avg_forgetting = sum([max_accuracy[i] - final_accuracies[i] for i in range(min(len(max_accuracy), len(final_accuracies)))]) / n_tasks
        writer.writerow(['Average', '', '', f"{avg_forgetting:.2f}"])
    
    # Save to text file
    summary_file = log_dir / "results_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Method: {method}\n")
        f.write(f"Number of tasks: {n_tasks}\n\n")
        
        f.write("Task-wise Results:\n")
        f.write("-" * 40 + "\n")
        
        for i in range(n_tasks):
            f.write(f"Task {i}:\n")
            f.write(f"  Max Accuracy: {max_accuracy[i]:.2f}%\n")
            if i < len(final_accuracies):
                f.write(f"  Final Accuracy: {final_accuracies[i]:.2f}%\n")
                f.write(f"  Forgetting: {max_accuracy[i] - final_accuracies[i]:.2f}%\n")
            f.write("\n")
        
        avg_forgetting = sum([max_accuracy[i] - final_accuracies[i] for i in range(min(len(max_accuracy), len(final_accuracies)))]) / n_tasks
        f.write(f"Average Forgetting: {avg_forgetting:.2f}%\n")