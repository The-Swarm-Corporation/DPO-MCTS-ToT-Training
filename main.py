# #!/usr/bin/env python3
"""
Production-grade DPO–Monte Carlo Tree of Thoughts example.

This module implements a post-training mechanism that allows a language model to
explore various reasoning branches (chain-of-thoughts) using a Monte Carlo Tree
Search (MCTS) framework. It selects the branch with the best answer using a cosine
similarity evaluator that compares the candidate answer to a known correct answer.

The language model uses Hugging Face's GPT-2 (via the text-generation pipeline)
and the evaluator uses a pretrained SentenceTransformer to compute sentence embeddings.
"""

import math
import random
from typing import List, Optional, Callable

import torch
from torch import nn
import torch.nn.functional as F
from loguru import logger

from transformers import pipeline, Pipeline
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# MCTS Node and Helper Functions
# ---------------------------------------------------------------------

class Node:
    """
    Represents a node in the MCTS tree.

    Attributes:
        state (List[str]): The chain-of-thought history up to this node.
        parent (Optional[Node]): The parent node in the tree.
        children (List[Node]): Child nodes generated from this node.
        visits (int): Number of visits for this node.
        total_score (float): Sum of evaluation scores from rollouts passing through this node.
        expanded (bool): Flag indicating whether the node has been expanded.
    """
    def __init__(self, state: List[str], parent: Optional["Node"] = None) -> None:
        self.state: List[str] = state
        self.parent: Optional["Node"] = parent
        self.children: List["Node"] = []
        self.visits: int = 0
        self.total_score: float = 0.0
        self.expanded: bool = False

    @property
    def average_score(self) -> float:
        """Returns the average evaluation score of this node."""
        return self.total_score / self.visits if self.visits > 0 else 0.0

    def is_terminal(self, terminal_condition: Callable[[List[str]], bool]) -> bool:
        """
        Checks whether the current state satisfies the terminal condition.
        
        Args:
            terminal_condition: Function that returns True if state is terminal.
            
        Returns:
            bool: True if terminal; otherwise False.
        """
        return terminal_condition(self.state)

    def add_child(self, child: "Node") -> None:
        """Adds a child node to the current node."""
        self.children.append(child)

    def update(self, score: float) -> None:
        """
        Updates the node's statistics using the provided score.
        
        Args:
            score (float): Evaluation score from a rollout.
        """
        self.visits += 1
        self.total_score += score


def ucb_value(child: Node, parent_visits: int, exploration_const: float) -> float:
    """
    Computes the Upper Confidence Bound (UCB) value for a child node.
    
    Args:
        child (Node): The child node.
        parent_visits (int): Total visits of the parent node.
        exploration_const (float): Exploration constant.
    
    Returns:
        float: UCB value.
    """
    if child.visits == 0:
        return float("inf")
    return child.average_score + exploration_const * math.sqrt(math.log(parent_visits) / child.visits)


def select_child_with_ucb(node: Node, exploration_const: float) -> Node:
    """
    Selects a child node from the given node based on the UCB value.
    
    Args:
        node (Node): The parent node.
        exploration_const (float): The exploration constant.
    
    Returns:
        Node: The selected child node.
    """
    best_child: Optional[Node] = None
    best_ucb: float = float("-inf")
    for child in node.children:
        current_ucb = ucb_value(child, node.visits, exploration_const)
        if current_ucb > best_ucb:
            best_ucb = current_ucb
            best_child = child
    if best_child is None:
        raise ValueError("No children available for UCB selection.")
    return best_child


def select_newly_added_child(node: Node) -> Node:
    """
    Selects one of the newly added (unvisited) children if available.
    
    Args:
        node (Node): The parent node.
    
    Returns:
        Node: A newly added child node.
    """
    unvisited = [child for child in node.children if child.visits == 0]
    if unvisited:
        return random.choice(unvisited)
    return random.choice(node.children)


def backpropagate(node: Node, score: float) -> None:
    """
    Backpropagates the evaluation score up the tree.
    
    Args:
        node (Node): The node from which to start backpropagation.
        score (float): The evaluation score.
    """
    current: Optional[Node] = node
    while current is not None:
        current.update(score)
        current = current.parent


# ---------------------------------------------------------------------
# Production Language Model Using Hugging Face Transformers
# ---------------------------------------------------------------------

class ProductionLanguageModel:
    """
    Production-grade language model interface using Hugging Face's GPT-2.

    Methods:
        generate_actions: Given a chain-of-thought state, returns candidate continuations.
        sample_action: Samples one candidate continuation.
    """
    def __init__(self, model_name: str = "gpt2", num_candidates: int = 3, max_new_tokens: int = 20) -> None:
        """
        Initializes the text generation pipeline.
        
        Args:
            model_name (str): Name of the pre-trained model.
            num_candidates (int): Number of candidate outputs per generation.
            max_new_tokens (int): Maximum tokens to generate.
        """
        self.num_candidates = num_candidates
        self.max_new_tokens = max_new_tokens
        # Initialize the text-generation pipeline (set to fast generation).
        self.generator: Pipeline = pipeline(
            "text-generation", model=model_name, tokenizer=model_name, framework="pt"
        )

    def generate_actions(self, state: List[str]) -> List[str]:
        """
        Generates candidate continuations from the current chain-of-thought state.
        
        Args:
            state (List[str]): The current chain-of-thought.
            
        Returns:
            List[str]: List of candidate continuations.
        """
        prompt = "\n".join(state) + "\n"
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.num_candidates,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=50256  # GPT-2's end-of-text token
            )
            # Extract the generated continuation after the prompt.
            actions = []
            for output in outputs:
                generated = output.get("generated_text", "")
                # Remove the original prompt from the generated text.
                continuation = generated[len(prompt):].strip()
                if continuation:
                    actions.append(continuation)
            return actions
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            return []

    def sample_action(self, state: List[str]) -> str:
        """
        Samples one candidate continuation from the current state.
        
        Args:
            state (List[str]): The current chain-of-thought.
            
        Returns:
            str: A single candidate continuation.
        """
        actions = self.generate_actions(state)
        if actions:
            return random.choice(actions)
        return "No continuation available."


# ---------------------------------------------------------------------
# Production Cosine Similarity Evaluator Using SentenceTransformers
# ---------------------------------------------------------------------

class CosineSimilarityEvaluator(nn.Module):
    """
    Evaluator that computes the cosine similarity between a candidate answer and the correct answer.
    
    This version uses a pretrained SentenceTransformer to generate sentence embeddings.
    """
    def __init__(self, correct_answer: str, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initializes the evaluator.
        
        Args:
            correct_answer (str): The known correct answer.
            model_name (str): The SentenceTransformer model name.
        """
        super().__init__()
        self.correct_answer = correct_answer
        # Load the SentenceTransformer model.
        self.encoder = SentenceTransformer(model_name)
        # Pre-compute and normalize the correct answer embedding.
        self.correct_embedding = self._get_normalized_embedding(correct_answer)

    def _get_normalized_embedding(self, text: str) -> torch.Tensor:
        """
        Generates a normalized embedding for the given text.
        
        Args:
            text (str): Input text.
            
        Returns:
            torch.Tensor: Normalized embedding tensor.
        """
        embedding = self.encoder.encode([text], convert_to_tensor=True)
        norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
        return embedding / norm

    def evaluate(self, answer: str) -> float:
        """
        Evaluates the candidate answer by computing its cosine similarity with the correct answer.
        
        Args:
            answer (str): Candidate answer text.
            
        Returns:
            float: Cosine similarity score between -1 and 1.
        """
        self.eval()
        with torch.no_grad():
            candidate_embedding = self.encoder.encode([answer], convert_to_tensor=True)
            norm = torch.norm(candidate_embedding, p=2, dim=1, keepdim=True)
            candidate_embedding = candidate_embedding / norm
            cosine_sim = F.cosine_similarity(candidate_embedding, self.correct_embedding, dim=1)
        score = cosine_sim.item()
        logger.debug(f"Cosine similarity for answer [{answer[:30]}...]: {score:.4f}")
        return score


# ---------------------------------------------------------------------
# DPO-MCTS Tree-of-Thought Search
# ---------------------------------------------------------------------

def dpo_mcts_tree_of_thought(
    prompt: str,
    model: ProductionLanguageModel,
    evaluator: CosineSimilarityEvaluator,
    budget: int,
    exploration_const: float,
    terminal_condition: Callable[[List[str]], bool],
    answer_extractor: Callable[[List[str]], str],
) -> str:
    """
    Runs the DPO-MCTS Tree-of-Thought algorithm.
    
    Args:
        prompt (str): The initial prompt.
        model (ProductionLanguageModel): The language model for generating continuations.
        evaluator (CosineSimilarityEvaluator): The evaluator for scoring answers.
        budget (int): Number of MCTS iterations.
        exploration_const (float): Exploration constant for UCB.
        terminal_condition (Callable): Function that returns True if a state is terminal.
        answer_extractor (Callable): Function to extract the final answer from the state.
    
    Returns:
        str: The final answer selected by the MCTS search.
    """
    logger.info("Starting DPO-MCTS Tree-of-Thought search.")
    root = Node(state=[prompt], parent=None)

    for iteration in range(budget):
        logger.info(f"Iteration {iteration + 1}/{budget}")
        node = root

        # --- Selection Phase ---
        while node.children and not node.is_terminal(terminal_condition):
            node = select_child_with_ucb(node, exploration_const)

        # --- Expansion Phase ---
        if not node.is_terminal(terminal_condition):
            actions = model.generate_actions(node.state)
            if actions:
                for action in actions:
                    new_state = node.state + [action]
                    child_node = Node(state=new_state, parent=node)
                    node.add_child(child_node)
                node.expanded = True
                node = select_newly_added_child(node)
            else:
                logger.warning("No actions generated during expansion.")

        # --- Simulation (Rollout) Phase ---
        simulation_state = list(node.state)
        rollout_steps = 0
        while not terminal_condition(simulation_state):
            rollout_action = model.sample_action(simulation_state)
            simulation_state.append(rollout_action)
            rollout_steps += 1
            if rollout_steps > 100:
                logger.warning("Rollout exceeded 100 steps; terminating rollout to avoid infinite loop.")
                break

        answer = answer_extractor(simulation_state)
        logger.info(f"Simulated answer: {answer}")

        # --- Evaluation Phase ---
        score = evaluator.evaluate(answer)
        logger.info(f"Evaluated cosine similarity score: {score:.4f}")

        # --- Backpropagation Phase ---
        backpropagate(node, score)

    # Final selection: choose the best child of the root by average score.
    if root.children:
        best_node = max(root.children, key=lambda n: n.average_score)
    else:
        best_node = root

    final_answer = answer_extractor(best_node.state)
    logger.info(f"Final answer selected: {final_answer}")
    return final_answer


# ---------------------------------------------------------------------
# Helper Functions for Terminal Condition and Answer Extraction
# ---------------------------------------------------------------------

def example_terminal_condition(state: List[str]) -> bool:
    """
    Example terminal condition: considers a state terminal if the last line ends with a period.
    
    Args:
        state (List[str]): The current chain-of-thought state.
        
    Returns:
        bool: True if the last element ends with a period; otherwise False.
    """
    if not state:
        return False
    return state[-1].strip().endswith(".")


def example_answer_extractor(state: List[str]) -> str:
    """
    Example answer extractor: returns the last element of the chain-of-thought.
    
    Args:
        state (List[str]): The chain-of-thought state.
        
    Returns:
        str: The final answer.
    """
    return state[-1] if state else ""


# ---------------------------------------------------------------------
# Main: Example Production Run
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Set up logging level (adjust as needed)
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    
    # Define prompt and known correct answer.
    initial_prompt = "What is the capital of France?"
    correct_answer = "Paris."  # Reference answer

    # Initialize the production language model (using GPT-2).
    language_model = ProductionLanguageModel(model_name="gpt2", num_candidates=3, max_new_tokens=20)

    # Initialize the cosine similarity evaluator (using a sentence transformer).
    evaluator = CosineSimilarityEvaluator(correct_answer=correct_answer, model_name="all-MiniLM-L6-v2")

    # Set MCTS parameters.
    iterations = 20
    exploration_parameter = 1.41

    # Run the DPO-MCTS Tree-of-Thought search.
    best_answer = dpo_mcts_tree_of_thought(
        prompt=initial_prompt,
        model=language_model,
        evaluator=evaluator,
        budget=iterations,
        exploration_const=exploration_parameter,
        terminal_condition=example_terminal_condition,
        answer_extractor=example_answer_extractor,
    )

    logger.info(f"\nBest answer found: {best_answer}")
