"""
⚡ PROMPT ENHANCER - Intelligent Query Processing System
═══════════════════════════════════════════════════════════════════════════════

This module enhances user prompts through multi-stage processing:
- Analyzes prompt intent and context
- Expands brief queries into comprehensive requests
- Adds quality directives naturally
- Shows real-time processing progress (1-100%)
- Outputs ultra-optimized prompts for maximum AI response

NO ENFORCEMENT - Uses natural flow for AI guidance
"""

import asyncio
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import re


class EnhancementStage(Enum):
    """Stages of prompt enhancement"""
    ANALYZING = "Analyzing intent..."
    UNDERSTANDING = "Understanding context..."
    EXPANDING = "Expanding query..."
    OPTIMIZING = "Optimizing for clarity..."
    ENHANCING = "Enhancing depth..."
    AMPLIFYING = "Amplifying intelligence..."
    FINALIZING = "Finalizing ultra-prompt..."


@dataclass
class EnhancementConfig:
    """Configuration for prompt enhancement"""
    show_progress: bool = True
    progress_speed: float = 0.015  # Seconds between updates (faster = 0.01)
    enhancement_depth: int = 5  # 1-5, higher = more enhancement
    add_quality_hints: bool = True
    add_depth_hints: bool = True
    add_speed_hints: bool = True
    natural_language: bool = True  # No enforcement, natural flow


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 NATURAL FLOW PROMPTS (No Enforcement)
# ═══════════════════════════════════════════════════════════════════════════════

NATURAL_QUALITY_HINTS = """
As you craft your response, naturally include:
• Comprehensive details that fully address the query
• Expert-level insights drawn from deep knowledge
• Practical examples that illuminate concepts
• Clear structure that guides understanding
• Depth that satisfies curiosity completely
"""

NATURAL_SPEED_HINTS = """
Your thoughts flow freely and quickly:
• Direct access to relevant knowledge
• Efficient expression of ideas
• No hesitation in sharing expertise
• Confident, clear communication
• Streamlined response generation
"""

NATURAL_DEPTH_HINTS = """
Your understanding runs deep:
• Multiple perspectives considered naturally
• Connections between concepts revealed
• Hidden insights brought to light  
• Complete picture painted
• Nothing important left unsaid
"""

ULTRA_ENHANCEMENT_TEMPLATE = """
╭──────────────────────────────────────────────────────────────╮
│              ✨ ENHANCED QUERY ✨                            │
╰──────────────────────────────────────────────────────────────╯

【ORIGINAL INTENT】
{original_query}

【EXPANDED UNDERSTANDING】
{expanded_context}

【WHAT THE USER TRULY WANTS】
{true_intent}

【COMPREHENSIVE REQUEST】
{comprehensive_request}

╭──────────────────────────────────────────────────────────────╮
│              🎯 RESPONSE GUIDANCE 🎯                         │
╰──────────────────────────────────────────────────────────────╯

{natural_hints}

Now, with complete understanding, provide an exceptional response.
"""


class PromptEnhancer:
    """
    ✨ INTELLIGENT PROMPT ENHANCEMENT SYSTEM
    
    Transforms simple user queries into ultra-optimized prompts
    through multi-stage processing with visual progress.
    
    Features:
    - Real-time progress display (1-100%)
    - Intent analysis and expansion
    - Natural quality guidance (no enforcement)
    - Context enrichment
    - Ultra-fast processing
    """
    
    def __init__(self, config: Optional[EnhancementConfig] = None):
        self.config = config or EnhancementConfig()
        self.enhancement_patterns = self._load_patterns()
        
    def _load_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load query type patterns for intelligent enhancement."""
        return {
            "code": {
                "keywords": ["code", "function", "class", "program", "script", "bug", "error", "implement", "write", "create", "fix", "python", "javascript", "api"],
                "expansion": "Provide complete, production-ready code with comprehensive error handling, detailed comments, and best practices. Include examples and edge case handling.",
                "depth": "Consider security, performance, maintainability, and extensibility."
            },
            "explain": {
                "keywords": ["explain", "what", "how", "why", "describe", "tell", "understand", "mean", "definition"],
                "expansion": "Provide a thorough, multi-level explanation with examples, analogies, and practical applications. Cover the topic from basics to advanced.",
                "depth": "Include historical context, current applications, and future implications."
            },
            "analyze": {
                "keywords": ["analyze", "compare", "evaluate", "assess", "review", "pros", "cons", "difference"],
                "expansion": "Conduct a comprehensive analysis considering all angles, with structured comparisons, evidence-based conclusions, and actionable insights.",
                "depth": "Include quantitative and qualitative factors, edge cases, and expert perspectives."
            },
            "create": {
                "keywords": ["create", "make", "build", "design", "generate", "write", "compose", "draft"],
                "expansion": "Create something exceptional, complete, and immediately usable. Include all necessary components and detailed documentation.",
                "depth": "Consider user experience, best practices, and innovative approaches."
            },
            "solve": {
                "keywords": ["solve", "fix", "help", "problem", "issue", "challenge", "stuck", "debug"],
                "expansion": "Provide a complete solution with step-by-step guidance, root cause analysis, and prevention strategies.",
                "depth": "Consider multiple solution approaches and their trade-offs."
            },
            "general": {
                "keywords": [],
                "expansion": "Provide a comprehensive, well-structured response that fully addresses the query with expert-level insight.",
                "depth": "Include relevant context, examples, and actionable information."
            }
        }
    
    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query for targeted enhancement."""
        query_lower = query.lower()
        
        for query_type, pattern in self.enhancement_patterns.items():
            if query_type == "general":
                continue
            for keyword in pattern["keywords"]:
                if keyword in query_lower:
                    return query_type
        
        return "general"
    
    def _expand_query(self, query: str, query_type: str) -> Dict[str, str]:
        """Expand the query with intelligent context."""
        pattern = self.enhancement_patterns[query_type]
        
        # Analyze original intent
        original_intent = query.strip()
        
        # Create expanded context
        expanded_context = f"""
The user is asking about: {original_intent}
Query type detected: {query_type.upper()}
This requires: {pattern['expansion']}
"""
        
        # Determine true intent
        true_intent = f"""
Beyond the literal question, the user wants:
• A complete, satisfying answer to their query
• Information they can immediately use
• Expert-level quality without gaps
• Clear, well-organized response
"""
        
        # Build comprehensive request
        comprehensive_request = f"""
{original_intent}

Additional context for optimal response:
• {pattern['expansion']}
• {pattern['depth']}
"""
        
        return {
            "original_query": original_intent,
            "expanded_context": expanded_context.strip(),
            "true_intent": true_intent.strip(),
            "comprehensive_request": comprehensive_request.strip()
        }
    
    def _build_natural_hints(self) -> str:
        """Build natural (non-enforcement) guidance."""
        hints = []
        
        if self.config.add_quality_hints:
            hints.append(NATURAL_QUALITY_HINTS)
        
        if self.config.add_speed_hints:
            hints.append(NATURAL_SPEED_HINTS)
            
        if self.config.add_depth_hints:
            hints.append(NATURAL_DEPTH_HINTS)
        
        return "\n".join(hints)
    
    async def enhance_with_progress(
        self,
        query: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> str:
        """
        Enhance query with visual progress display.
        
        Args:
            query: Original user query
            progress_callback: Optional callback for progress updates
            
        Returns:
            Ultra-enhanced prompt
        """
        stages = [
            (10, EnhancementStage.ANALYZING),
            (25, EnhancementStage.UNDERSTANDING),
            (45, EnhancementStage.EXPANDING),
            (65, EnhancementStage.OPTIMIZING),
            (80, EnhancementStage.ENHANCING),
            (95, EnhancementStage.AMPLIFYING),
            (100, EnhancementStage.FINALIZING),
        ]
        
        # Process through stages
        for progress, stage in stages:
            if progress_callback:
                progress_callback(progress, stage.value)
            await asyncio.sleep(self.config.progress_speed)
        
        # Actually enhance the prompt
        return self.enhance(query)
    
    def enhance(self, query: str) -> str:
        """
        Enhance a query without progress display.
        
        Args:
            query: Original user query
            
        Returns:
            Ultra-enhanced prompt
        """
        # Detect query type
        query_type = self._detect_query_type(query)
        
        # Expand the query
        expansion = self._expand_query(query, query_type)
        
        # Build natural hints
        natural_hints = self._build_natural_hints()
        
        # Create the enhanced prompt
        enhanced = ULTRA_ENHANCEMENT_TEMPLATE.format(
            original_query=expansion["original_query"],
            expanded_context=expansion["expanded_context"],
            true_intent=expansion["true_intent"],
            comprehensive_request=expansion["comprehensive_request"],
            natural_hints=natural_hints
        )
        
        return enhanced
    
    def quick_enhance(self, query: str) -> str:
        """
        Fast enhancement without full template - for speed.
        
        Args:
            query: Original query
            
        Returns:
            Quickly enhanced prompt
        """
        query_type = self._detect_query_type(query)
        pattern = self.enhancement_patterns[query_type]
        
        return f"""
【Query】{query}

【Context】{pattern['expansion']}

【Depth】{pattern['depth']}

Respond with your full capability - comprehensive, detailed, exceptional quality.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# ⚡ ULTRA-FAST NATURAL SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

ULTRA_NATURAL_SYSTEM_PROMPT = """
╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮
│                      ✨ NEXUS AI - NATURAL FLOW MODE ✨                      │
╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯

You are an extraordinarily capable AI, naturally operating at peak performance.
Your responses flow from deep knowledge and genuine capability.

┌─────────────────────────────────────────────────────────────────────────────┐
│                         💫 YOUR NATURAL STATE 💫                            │
└─────────────────────────────────────────────────────────────────────────────┘

You naturally possess:
• Vast, instantly accessible knowledge across all domains
• The combined expertise of countless specialists
• Deep analytical and creative capabilities  
• Clear, efficient communication abilities
• Genuine desire to provide exceptional help

┌─────────────────────────────────────────────────────────────────────────────┐
│                         🌊 NATURAL RESPONSE FLOW 🌊                          │
└─────────────────────────────────────────────────────────────────────────────┘

When responding, you naturally:
• Share knowledge freely and completely
• Provide rich, detailed explanations
• Include practical examples and insights
• Structure information for clarity
• Give your absolute best on every query

There's no need to hold back or simplify - your natural state is excellence.
Brief queries receive comprehensive answers because you understand the deeper need.

┌─────────────────────────────────────────────────────────────────────────────┐
│                         ⚡ SPEED & QUALITY ⚡                                │
└─────────────────────────────────────────────────────────────────────────────┘

Speed and quality flow together naturally:
• Knowledge retrieval is instant
• Expression is clear and efficient  
• Depth doesn't slow you down
• Excellence is your default state
• Every response demonstrates mastery

┌─────────────────────────────────────────────────────────────────────────────┐
│                         🎯 YOUR APPROACH 🎯                                  │
└─────────────────────────────────────────────────────────────────────────────┘

For every query, you naturally:
1. Fully understand the underlying need
2. Access all relevant knowledge
3. Craft a comprehensive, excellent response
4. Include insights that add exceptional value
5. Present information with perfect clarity

This isn't effort - it's simply who you are.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                   ✨ NATURAL EXCELLENCE ACTIVATED ✨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 PROGRESS BAR GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def create_progress_bar(progress: int, width: int = 40) -> str:
    """Create a visual progress bar."""
    filled = int(width * progress / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {progress}%"


def create_fancy_progress(progress: int, stage: str) -> str:
    """Create a fancy progress display."""
    bar = create_progress_bar(progress)
    
    if progress < 25:
        color = "cyan"
        emoji = "🔍"
    elif progress < 50:
        color = "blue"
        emoji = "⚡"
    elif progress < 75:
        color = "yellow"
        emoji = "🚀"
    elif progress < 100:
        color = "magenta"
        emoji = "✨"
    else:
        color = "green"
        emoji = "🔥"
    
    return f"{emoji} {bar} {stage}"


# Global instance
_prompt_enhancer: Optional[PromptEnhancer] = None


def get_prompt_enhancer() -> PromptEnhancer:
    """Get the global prompt enhancer."""
    global _prompt_enhancer
    if _prompt_enhancer is None:
        _prompt_enhancer = PromptEnhancer()
    return _prompt_enhancer


def enhance_prompt(query: str) -> str:
    """Quick function to enhance a prompt."""
    return get_prompt_enhancer().enhance(query)


async def enhance_prompt_with_progress(
    query: str,
    callback: Optional[Callable[[int, str], None]] = None
) -> str:
    """Enhance prompt with progress display."""
    return await get_prompt_enhancer().enhance_with_progress(query, callback)


def get_natural_system_prompt() -> str:
    """Get the natural (non-enforcement) system prompt."""
    return ULTRA_NATURAL_SYSTEM_PROMPT
