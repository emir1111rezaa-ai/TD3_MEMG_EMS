#!/usr/bin/env python3
"""
TD3 Agent for Multi-Energy Microgrid (MEMG) Energy Management System

Main entry point for training and evaluation

Usage:
    python main.py --train --episodes 50
    python main.py --eval --checkpoint ./checkpoints/td3_episode_50.pt
"""

import argparse
import torch
from pathlib import Path
from training.trainer import TD3Trainer
from training.config import Config


def main():
    parser = argparse.ArgumentParser(
        description='TD3 Agent for MEMG Energy Management'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the TD3 agent'
    )
    
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Evaluate trained agent'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=50,
        help='Number of training episodes (default: 50)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to load'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device to use (default: cpu)'
    )
    
    parser.add_argument(
        '--save-config',
        action='store_true',
        help='Save configuration to YAML file'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*70)
    print("TD3 Agent for Multi-Energy Microgrid (MEMG)")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("="*70 + "\n")
    
    # Save configuration if requested
    if args.save_config:
        Config.save_config('config.yaml')
    
    # Create trainer
    trainer = TD3Trainer(
        state_dim=Config.AGENT_CONFIG['state_dim'],
        action_dim=Config.AGENT_CONFIG['action_dim'],
        hidden_dim=Config.AGENT_CONFIG['hidden_dim'],
        batch_size=Config.AGENT_CONFIG['batch_size'],
        device=args.device,
        save_dir=Config.TRAINING_CONFIG['save_dir']
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            trainer.load_checkpoint(str(checkpoint_path))
        else:
            print(f"Warning: Checkpoint file not found: {args.checkpoint}")
    
    # Training phase
    if args.train:
        print(f"\nStarting training with {args.episodes} episodes...\n")
        trainer.train(
            num_episodes=args.episodes,
            eval_interval=Config.TRAINING_CONFIG['eval_interval'],
            checkpoint_interval=Config.TRAINING_CONFIG['checkpoint_interval']
        )
        
        # Evaluate after training
        print("\nEvaluating trained agent...")
        eval_results = trainer.evaluate(num_episodes=Config.TRAINING_CONFIG['eval_episodes'])
        
        print("\n" + "="*70)
        print("Evaluation Results:")
        print("="*70)
        print(f"Mean Reward: {eval_results['mean_reward']:.2f}")
        print(f"Mean Cost: {eval_results['mean_cost']:.2f} €")
        print(f"Std Cost: {eval_results['std_cost']:.2f} €")
        print(f"Max Cost: {eval_results['max_cost']:.2f} €")
        print(f"Mean Constraint Violations: {eval_results['mean_violations']:.1f}")
        print("="*70 + "\n")
    
    # Evaluation phase (without training)
    elif args.eval:
        if not args.checkpoint:
            print("Error: --checkpoint required for --eval")
            return
        
        print(f"\nEvaluating agent from checkpoint: {args.checkpoint}\n")
        eval_results = trainer.evaluate(num_episodes=Config.TRAINING_CONFIG['eval_episodes'])
        
        print("\n" + "="*70)
        print("Evaluation Results:")
        print("="*70)
        print(f"Mean Reward: {eval_results['mean_reward']:.2f}")
        print(f"Mean Cost: {eval_results['mean_cost']:.2f} €")
        print(f"Std Cost: {eval_results['std_cost']:.2f} €")
        print(f"Max Cost: {eval_results['max_cost']:.2f} €")
        print(f"Mean Constraint Violations: {eval_results['mean_violations']:.1f}")
        print("="*70 + "\n")
    
    else:
        print("Please use --train or --eval flag")
        print("Usage examples:")
        print("  python main.py --train --episodes 50")
        print("  python main.py --eval --checkpoint ./checkpoints/td3_episode_50.pt")


if __name__ == '__main__':
    main()
