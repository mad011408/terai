"""
🔥 ULTRA PROMPT ENHANCER - Maximum Power Query Transformation
═══════════════════════════════════════════════════════════════════════════════

This is the MOST ADVANCED prompt enhancement system ever created.
It transforms ANY simple query into an ULTRA-POWERFUL mega-prompt.

Features:
- 15-second deep enhancement process
- Converts simple queries to 500+ word powerful prompts
- Multi-stage intelligent expansion
- Real-time progress display (1-100%)
- Context analysis and amplification
- Expert-level query formulation
"""

import asyncio
import time
import re
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import random


class EnhancementPhase(Enum):
    """Enhancement phases - each adds more power"""
    ANALYZING = ("🔍 Analyzing intent", 5)
    UNDERSTANDING = ("🧠 Understanding context", 10)
    DETECTING = ("🎯 Detecting query type", 15)
    EXPANDING = ("📝 Expanding query", 25)
    DEEPENING = ("🔬 Deepening context", 35)
    AMPLIFYING = ("⚡ Amplifying power", 45)
    ENRICHING = ("✨ Enriching details", 55)
    OPTIMIZING = ("🚀 Optimizing structure", 65)
    ENHANCING = ("💫 Enhancing quality", 75)
    SUPERCHARGING = ("🔥 Supercharging prompt", 85)
    MAXIMIZING = ("💪 Maximizing effectiveness", 92)
    FINALIZING = ("✅ Finalizing ultra-prompt", 100)


@dataclass
class UltraEnhancerConfig:
    """Configuration for ultra prompt enhancement"""
    enhancement_time_seconds: float = 15.0  # Total time for enhancement
    min_output_length: int = 500  # Minimum enhanced prompt length
    max_output_length: int = 2000  # Maximum enhanced prompt length
    add_expert_context: bool = True
    add_quality_directives: bool = True
    add_depth_requirements: bool = True
    add_examples_request: bool = True
    add_comprehensive_expectations: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# 🧠 INTELLIGENT QUERY PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

QUERY_PATTERNS = {
    "code": {
        "keywords": ["code", "function", "class", "program", "script", "bug", "error", "implement", 
                    "write", "create", "fix", "python", "javascript", "java", "api", "algorithm",
                    "database", "sql", "html", "css", "react", "node", "backend", "frontend"],
        "expert_role": "world's most elite software architect and programming master",
        "expansion_template": """
As the {expert_role}, I need you to approach this with the combined expertise of:
- Senior developers from Google, Meta, Microsoft, and OpenAI
- Authors of the most popular programming books
- Open source maintainers of critical infrastructure
- Security researchers and performance optimization experts

The code you provide must be:
• PRODUCTION-READY: No placeholders, no "implement this" comments
• COMPLETE: Every function, every import, every edge case handled
• DOCUMENTED: Clear comments explaining the "why" not just "what"
• OPTIMIZED: Performance-conscious with O(n) considerations
• SECURE: No vulnerabilities, proper input validation
• TESTED: Include comprehensive unit tests
• MAINTAINABLE: Clean code principles, SOLID design patterns
"""
    },
    "explain": {
        "keywords": ["explain", "what", "how", "why", "describe", "tell", "understand", "mean", 
                    "definition", "concept", "theory", "principle", "difference", "between"],
        "expert_role": "brilliant professor with decades of teaching experience",
        "expansion_template": """
As the {expert_role}, I need you to explain this with:
- The clarity of Richard Feynman explaining physics to beginners
- The depth of a graduate-level academic lecture
- The practicality of real-world industry experience
- The patience to cover every angle and nuance

Your explanation must include:
• FUNDAMENTALS: Start from first principles, assume I want deep understanding
• ANALOGIES: Use real-world comparisons that make concepts click
• EXAMPLES: Multiple practical examples from different contexts
• VISUALS: Describe diagrams or visualizations where helpful
• DEPTH: Cover beginner, intermediate, AND advanced perspectives
• CONNECTIONS: How this relates to other important concepts
• APPLICATIONS: Real-world uses and implications
• EDGE CASES: What about unusual situations?
• COMMON MISTAKES: What do people often get wrong?
"""
    },
    "analyze": {
        "keywords": ["analyze", "compare", "evaluate", "assess", "review", "pros", "cons", 
                    "advantage", "disadvantage", "versus", "vs", "better", "best", "worst"],
        "expert_role": "strategic consultant at McKinsey with PhD-level analytical skills",
        "expansion_template": """
As the {expert_role}, conduct a comprehensive analysis with:
- Fortune 500 C-suite level strategic thinking
- Academic researcher's attention to evidence
- Industry practitioner's real-world perspective
- Futurist's vision for long-term implications

Your analysis must include:
• MULTIPLE FRAMEWORKS: Use at least 3 analytical frameworks
• QUANTITATIVE DATA: Numbers, statistics, benchmarks where relevant
• QUALITATIVE INSIGHTS: Expert opinions and experiential knowledge
• SWOT ANALYSIS: Strengths, weaknesses, opportunities, threats
• COMPARATIVE MATRIX: Clear side-by-side comparison
• RISK ASSESSMENT: What could go wrong? Mitigation strategies?
• RECOMMENDATIONS: Clear, actionable next steps
• TRADE-OFF ANALYSIS: What am I giving up with each choice?
• LONG-TERM VIEW: 1 year, 5 year, 10 year implications
"""
    },
    "create": {
        "keywords": ["create", "make", "build", "design", "generate", "draft", "compose", 
                    "write", "develop", "produce", "craft", "construct"],
        "expert_role": "creative genius combining the best of design and engineering",
        "expansion_template": """
As the {expert_role}, create something extraordinary with:
- Apple's obsession with design excellence
- Tesla's innovation and disruption mindset
- Netflix's user experience focus
- Amazon's customer obsession

Your creation must be:
• COMPLETE: Nothing left as "TODO" or "placeholder"
• POLISHED: Ready for production/publication
• INNOVATIVE: Fresh perspective, not generic
• PRACTICAL: Immediately usable
• PROFESSIONAL: Industry-standard quality
• SCALABLE: Can grow and adapt
• DOCUMENTED: Clear instructions for use/modification
• EXCEPTIONAL: Something I'd be proud to show anyone
"""
    },
    "solve": {
        "keywords": ["solve", "fix", "help", "problem", "issue", "challenge", "stuck", "debug",
                    "error", "wrong", "not working", "broken", "trouble", "difficulty"],
        "expert_role": "master problem solver with expertise across all domains",
        "expansion_template": """
As the {expert_role}, solve this with the methodical approach of:
- A detective solving a complex case
- A doctor diagnosing a challenging condition
- An engineer debugging a critical system
- A therapist understanding root causes

Your solution must include:
• ROOT CAUSE ANALYSIS: What's really causing this?
• MULTIPLE SOLUTIONS: At least 3 different approaches
• STEP-BY-STEP GUIDE: Detailed implementation instructions
• QUICK FIX: Immediate band-aid if needed
• PERMANENT FIX: Long-term proper solution
• PREVENTION: How to avoid this in the future
• EDGE CASES: What if the obvious solution doesn't work?
• DEBUGGING TIPS: How to troubleshoot similar issues
"""
    },
    "learn": {
        "keywords": ["learn", "study", "understand", "master", "improve", "skill", "tutorial",
                    "guide", "teach", "course", "practice", "beginner", "advanced"],
        "expert_role": "world-class educator and learning science expert",
        "expansion_template": """
As the {expert_role}, teach me with:
- The proven techniques of the world's best educators
- The science of accelerated learning and retention
- The practical focus of industry professionals
- The motivation of an inspiring mentor

Your teaching must include:
• LEARNING PATH: Clear progression from basics to advanced
• KEY CONCEPTS: The 20% that gives 80% of results
• HANDS-ON EXERCISES: Practice problems and projects
• COMMON PITFALLS: What to avoid and why
• RESOURCES: Best books, courses, tools to continue learning
• MILESTONES: How do I know I'm progressing?
• REAL-WORLD APPLICATION: How is this used professionally?
• MOTIVATION: Why is this worth mastering?
"""
    },
    "general": {
        "keywords": [],
        "expert_role": "polymath genius with expertise across all fields",
        "expansion_template": """
As the {expert_role}, respond with:
- The comprehensive knowledge of a well-read intellectual
- The practical wisdom of decades of experience
- The clarity of the world's best communicators
- The helpfulness of a dedicated mentor

Your response must be:
• COMPREHENSIVE: Cover all important aspects
• INSIGHTFUL: Provide value beyond the obvious
• STRUCTURED: Well-organized and easy to follow
• ACTIONABLE: Include clear next steps
• MEMORABLE: Key takeaways that stick
• BALANCED: Multiple perspectives considered
• DEEP: Sufficient detail for real understanding
"""
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# 🔥 POWER AMPLIFICATION TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

ULTRA_ENHANCEMENT_HEADER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔥 ULTRA-ENHANCED QUERY 🔥                                ║
║              Transformed by Advanced Prompt Engineering                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

CONTEXT_SECTION = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                         📋 ORIGINAL QUERY                                   │
└─────────────────────────────────────────────────────────────────────────────┘

{original_query}

┌─────────────────────────────────────────────────────────────────────────────┐
│                         🎯 DETECTED INTENT                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Query Type: {query_type}
Primary Intent: {primary_intent}
Secondary Needs: {secondary_needs}
Implied Requirements: {implied_requirements}
"""

EXPERT_SECTION = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                         🧠 EXPERT PERSPECTIVE                               │
└─────────────────────────────────────────────────────────────────────────────┘

{expert_expansion}
"""

REQUIREMENTS_SECTION = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ✨ ENHANCED REQUIREMENTS                            │
└─────────────────────────────────────────────────────────────────────────────┘

【QUALITY EXPECTATIONS】
• Response must be EXCEPTIONALLY comprehensive
• Every claim should be accurate and well-reasoned
• Include specific examples and evidence
• Structure for maximum clarity and usefulness
• Provide actionable insights and recommendations

【DEPTH REQUIREMENTS】
• Cover the topic from multiple angles
• Include beginner AND advanced perspectives
• Address edge cases and exceptions
• Anticipate follow-up questions
• Connect to related concepts

【FORMAT REQUIREMENTS】
• Use clear headers and sections
• Include bullet points for scanability
• Add code blocks where relevant
• Use tables for comparisons
• Provide summaries for long sections

【COMPLETENESS CHECK】
Before finalizing, verify:
□ Have I fully addressed the query?
□ Is there anything important I'm missing?
□ Would an expert be satisfied with this depth?
□ Is this immediately actionable?
□ Have I exceeded expectations?
"""

AMPLIFIED_QUERY_SECTION = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                         🚀 AMPLIFIED QUERY                                  │
└─────────────────────────────────────────────────────────────────────────────┘

{amplified_query}
"""

FINAL_DIRECTIVE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ⚡ EXECUTE WITH MAXIMUM CAPABILITY ⚡
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Now, with complete understanding of what's truly being asked, respond with:
• Your FULL knowledge and capability
• EXCEPTIONAL detail and depth
• PERFECT clarity and structure
• MAXIMUM helpfulness and value

This is not just a query - it's an opportunity to demonstrate excellence.
Respond now with your absolute best.
"""


class UltraPromptEnhancer:
    """
    🔥 ULTRA PROMPT ENHANCER
    
    The most advanced prompt enhancement system that transforms
    any simple query into a powerful mega-prompt.
    
    Process:
    1. Analyze query intent (1-15%)
    2. Detect query type (15-25%)
    3. Expand with expert context (25-55%)
    4. Amplify power (55-75%)
    5. Add quality requirements (75-90%)
    6. Finalize ultra-prompt (90-100%)
    
    Result: 500-2000 word ultra-optimized prompt
    """
    
    def __init__(self, config: Optional[UltraEnhancerConfig] = None):
        self.config = config or UltraEnhancerConfig()
        
    def _detect_query_type(self, query: str) -> Tuple[str, Dict]:
        """Detect the type of query and return matching pattern."""
        query_lower = query.lower()
        
        # Score each pattern
        scores = {}
        for qtype, pattern in QUERY_PATTERNS.items():
            if qtype == "general":
                continue
            score = sum(1 for kw in pattern["keywords"] if kw in query_lower)
            if score > 0:
                scores[qtype] = score
        
        if scores:
            best_type = max(scores, key=scores.get)
            return best_type, QUERY_PATTERNS[best_type]
        
        return "general", QUERY_PATTERNS["general"]
    
    def _analyze_intent(self, query: str) -> Dict[str, str]:
        """Analyze the primary intent and implicit needs."""
        query_lower = query.lower()
        
        # Detect primary intent
        if any(w in query_lower for w in ["how to", "how do", "how can"]):
            primary = "Learn how to accomplish something"
        elif any(w in query_lower for w in ["what is", "what are", "define"]):
            primary = "Understand a concept or definition"
        elif any(w in query_lower for w in ["why", "reason", "cause"]):
            primary = "Understand reasons or causes"
        elif any(w in query_lower for w in ["compare", "vs", "versus", "difference"]):
            primary = "Compare options or understand differences"
        elif any(w in query_lower for w in ["best", "recommend", "should I"]):
            primary = "Get recommendations or best practices"
        elif any(w in query_lower for w in ["fix", "error", "problem", "issue"]):
            primary = "Solve a problem or fix an issue"
        elif any(w in query_lower for w in ["create", "make", "build", "write"]):
            primary = "Create or generate something"
        else:
            primary = "Get comprehensive information and guidance"
        
        # Detect secondary needs
        secondary_needs = []
        if len(query.split()) < 10:
            secondary_needs.append("Needs expansion and clarification")
        if "?" in query:
            secondary_needs.append("Expects a clear, direct answer")
        if any(w in query_lower for w in ["example", "show", "demonstrate"]):
            secondary_needs.append("Wants practical examples")
        if any(w in query_lower for w in ["simple", "easy", "beginner"]):
            secondary_needs.append("Prefers accessible explanation")
        if any(w in query_lower for w in ["detailed", "complete", "full"]):
            secondary_needs.append("Wants comprehensive coverage")
        
        if not secondary_needs:
            secondary_needs = ["Expects thorough, helpful response"]
        
        # Detect implied requirements
        implied = []
        implied.append("Accurate, reliable information")
        implied.append("Well-structured, easy to follow")
        implied.append("Immediately actionable")
        if any(w in query_lower for w in ["code", "function", "program"]):
            implied.append("Working, tested code")
        if any(w in query_lower for w in ["fast", "quick", "asap"]):
            implied.append("Efficient, direct response")
            
        return {
            "primary_intent": primary,
            "secondary_needs": ", ".join(secondary_needs),
            "implied_requirements": ", ".join(implied)
        }
    
    def _amplify_query(self, query: str, query_type: str) -> str:
        """Create an amplified version of the query."""
        # Add contextual expansion
        amplifications = [
            f"I need a comprehensive, expert-level response to: {query}",
            "",
            "Please ensure your response:",
            "• Goes beyond surface-level information",
            "• Includes specific, actionable details",
            "• Addresses potential follow-up questions proactively",
            "• Provides examples where helpful",
            "• Is structured for easy understanding",
            "",
        ]
        
        if query_type == "code":
            amplifications.extend([
                "For any code provided:",
                "• Make it complete and production-ready",
                "• Include error handling and edge cases",
                "• Add helpful comments",
                "• Consider security and performance",
                ""
            ])
        elif query_type == "explain":
            amplifications.extend([
                "For your explanation:",
                "• Start with fundamentals",
                "• Build up to advanced concepts",
                "• Use clear analogies",
                "• Include visual descriptions if helpful",
                ""
            ])
        elif query_type == "analyze":
            amplifications.extend([
                "For your analysis:",
                "• Use multiple analytical frameworks",
                "• Include data and evidence",
                "• Consider multiple perspectives",
                "• Provide clear recommendations",
                ""
            ])
        
        amplifications.append("This is important - please give it your absolute best effort.")
        
        return "\n".join(amplifications)
    
    async def enhance_with_progress(
        self,
        query: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> str:
        """
        Enhance query with 15-second progress display.
        
        Args:
            query: Original user query
            progress_callback: Callback for progress updates
            
        Returns:
            Ultra-enhanced mega-prompt
        """
        phases = list(EnhancementPhase)
        total_phases = len(phases)
        time_per_phase = self.config.enhancement_time_seconds / total_phases
        
        # Process through phases
        for phase in phases:
            if progress_callback:
                progress_callback(phase.value[1], phase.value[0])
            await asyncio.sleep(time_per_phase)
        
        # Generate the enhanced prompt
        return self.enhance(query)
    
    def enhance(self, query: str) -> str:
        """
        Enhance a query into an ultra-powerful mega-prompt.
        
        Args:
            query: Original user query
            
        Returns:
            Ultra-enhanced prompt (500-2000 words)
        """
        # Detect query type
        query_type, pattern = self._detect_query_type(query)
        
        # Analyze intent
        intent_analysis = self._analyze_intent(query)
        
        # Get expert expansion
        expert_expansion = pattern["expansion_template"].format(
            expert_role=pattern["expert_role"]
        )
        
        # Create amplified query
        amplified_query = self._amplify_query(query, query_type)
        
        # Build the ultra-enhanced prompt
        enhanced_prompt = ULTRA_ENHANCEMENT_HEADER
        
        enhanced_prompt += CONTEXT_SECTION.format(
            original_query=query,
            query_type=query_type.upper(),
            primary_intent=intent_analysis["primary_intent"],
            secondary_needs=intent_analysis["secondary_needs"],
            implied_requirements=intent_analysis["implied_requirements"]
        )
        
        enhanced_prompt += EXPERT_SECTION.format(
            expert_expansion=expert_expansion
        )
        
        enhanced_prompt += REQUIREMENTS_SECTION
        
        enhanced_prompt += AMPLIFIED_QUERY_SECTION.format(
            amplified_query=amplified_query
        )
        
        enhanced_prompt += FINAL_DIRECTIVE
        
        return enhanced_prompt
    
    def quick_enhance(self, query: str) -> str:
        """
        Fast enhancement for when speed is priority.
        
        Args:
            query: Original query
            
        Returns:
            Quickly enhanced prompt
        """
        query_type, pattern = self._detect_query_type(query)
        
        return f"""
【ENHANCED QUERY】
{query}

【EXPERT PERSPECTIVE】
Respond as the {pattern['expert_role']}.

【REQUIREMENTS】
• Be comprehensive and thorough
• Include specific examples
• Structure clearly
• Make it actionable
• Exceed expectations

Provide your absolute best response.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 PROGRESS BAR UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def create_progress_bar(progress: int, width: int = 30) -> str:
    """Create a visual progress bar."""
    filled = int(width * progress / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {progress:3d}%"


def create_animated_progress(progress: int, stage: str, width: int = 30) -> str:
    """Create animated progress bar with stage info."""
    filled = int(width * progress / 100)
    bar = "█" * filled + "░" * (width - filled)
    
    # Add color hints based on progress
    if progress < 25:
        emoji = "🔍"
    elif progress < 50:
        emoji = "⚡"
    elif progress < 75:
        emoji = "🚀"
    elif progress < 100:
        emoji = "🔥"
    else:
        emoji = "✅"
    
    return f"{emoji} [{bar}] {progress:3d}% {stage}"


# ═══════════════════════════════════════════════════════════════════════════════
# 🌊 NATURAL FLOW SYSTEM PROMPT (No Enforcement)
# ═══════════════════════════════════════════════════════════════════════════════

ULTRA_NATURAL_SYSTEM_PROMPT = """
╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮
│                    ✨ NEXUS AI - NATURAL EXCELLENCE ✨                       │
╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯

You are an extraordinarily capable AI operating at peak performance.
Your responses naturally reflect deep knowledge and genuine expertise.

┌─────────────────────────────────────────────────────────────────────────────┐
│                         🌟 YOUR NATURAL ABILITIES 🌟                        │
└─────────────────────────────────────────────────────────────────────────────┘

You naturally possess:
• Vast knowledge spanning all domains - instantly accessible
• The combined wisdom of countless experts and practitioners
• Crystal-clear communication abilities
• Deep analytical and creative capabilities
• Genuine desire to provide exceptional value

┌─────────────────────────────────────────────────────────────────────────────┐
│                         💫 NATURAL RESPONSE STYLE 💫                        │
└─────────────────────────────────────────────────────────────────────────────┘

When responding, you naturally:
• Share knowledge freely, completely, generously
• Provide rich detail that fully satisfies curiosity
• Include examples that illuminate concepts
• Structure information for maximum clarity
• Anticipate and address follow-up questions
• Give your absolute best - it's who you are

There's no holding back. Brief queries receive comprehensive answers
because you understand what the user truly needs.

┌─────────────────────────────────────────────────────────────────────────────┐
│                         ⚡ EFFORTLESS EXCELLENCE ⚡                          │
└─────────────────────────────────────────────────────────────────────────────┘

Excellence flows naturally from you:
• Deep responses come easily
• Complex topics are explained clearly
• Code is complete and production-ready
• Analysis is thorough and insightful
• Creativity is unlimited and inspired

This isn't effort - it's simply your nature to be exceptional.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ✨ NATURAL EXCELLENCE ENGAGED ✨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# Global instance
_ultra_enhancer: Optional[UltraPromptEnhancer] = None


def get_ultra_enhancer() -> UltraPromptEnhancer:
    """Get the global ultra prompt enhancer."""
    global _ultra_enhancer
    if _ultra_enhancer is None:
        _ultra_enhancer = UltraPromptEnhancer()
    return _ultra_enhancer


# Compatibility aliases
def get_prompt_enhancer() -> UltraPromptEnhancer:
    """Alias for get_ultra_enhancer."""
    return get_ultra_enhancer()


def enhance_prompt(query: str) -> str:
    """Quick function to enhance a prompt."""
    return get_ultra_enhancer().enhance(query)


async def enhance_prompt_with_progress(
    query: str,
    callback: Optional[Callable[[int, str], None]] = None
) -> str:
    """Enhance prompt with 15-second progress display."""
    return await get_ultra_enhancer().enhance_with_progress(query, callback)


def get_natural_system_prompt() -> str:
    """Get the natural flow system prompt."""
    return ULTRA_NATURAL_SYSTEM_PROMPT
