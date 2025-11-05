// ============================================================================
// Pokemon Prompt Parser
// ============================================================================
// This BAML function parses natural language prompts into structured
// attributes for conditioning the Pokemon CVAE model.
// ============================================================================

// ----------------------------------------------------------------------------
// Parsed Prompt Attributes Class
// ----------------------------------------------------------------------------

class ParsedPromptAttributes {
  // Pokemon types
  type1 PokemonType          // Primary type (required)
  type2 PokemonType?         // Secondary type (optional, null if single-type)

  // Colors
  primary_color PokemonColor       // Dominant color (required)
  secondary_color PokemonColor?    // Secondary color (optional, null if single-color)

  // Body shape
  body_shape BodyShape             // Overall body structure (required)

  // Size
  size_class SizeClass             // Size category (required)

  // Evolution
  evolution_stage EvolutionStage   // Evolution stage (required)

  // Environment
  habitat Habitat                  // Natural habitat (required)

  // Special status
  is_legendary bool                // Is legendary Pokemon?
  is_mythical bool                 // Is mythical Pokemon?
}

// ----------------------------------------------------------------------------
// BAML Function - Parse Pokemon Prompt
// ----------------------------------------------------------------------------

function ParsePokemonPrompt(user_prompt: string) -> ParsedPromptAttributes {
  client CustomGPT5Mini  // Uses OpenAI GPT-5-Mini - fast and reliable

  prompt #"
    {{_.role("system")}}
    You are a Pokemon attribute parser for a ML generation system trained on real Pokemon metadata. Parse natural language prompts into structured attributes matching the training data format.

    TRAINING DATA PATTERNS (examples from dataset):
    - "Create a dragon-like Pokemon with fiery orange scales and majestic blue wings"
    - "Show me a water-type Pokemon with a shell and powerful cannons on its back"
    - "Generate a cute electric-type Pokemon that can generate powerful shocks"
    - "Design a dark-type Pokemon resembling a vulture with a strong presence in mountainous areas"
    - "Imagine a psychic-type Pokemon with a sleek design and a powerful aura"

    CRITICAL RULES:
    1. Extract ALL mentioned attributes explicitly
    2. For missing attributes, infer logically from context and Pokemon conventions
    3. Prioritize visual descriptors over abstract concepts
    4. Match the training data's level of detail

    TYPE INFERENCE (with context):
    - Explicit: "fire-type", "water-type", "electric-type", "psychic-type", "dark-type", "steel-type"
    - From abilities: "breathes flames" → Fire, "generate shocks" → Electric, "swim swiftly" → Water
    - From appearance: "metallic feathers" → Steel, "dragon-like" → Dragon, "bird" → Flying
    - From personality: "cunning" + "bird" → Dark/Flying
    - Dual types: look for TWO distinct type indicators (e.g., "fire-breathing dragon" → Fire/Dragon)

    COLOR INFERENCE (prioritize explicit mentions):
    - Explicit: "orange scales", "blue wings", "yellow fur", "black feathers", "purple body"
    - From descriptors: "fiery" → Red/Orange, "metallic" → Gray, "sleek dark" → Black
    - Type-based defaults:
      * Fire → Red + Orange secondary
      * Water → Blue + White secondary
      * Grass → Green + Yellow secondary
      * Electric → Yellow + Black/Red secondary
      * Psychic → Purple + White secondary
      * Steel/Rock → Gray + Brown secondary
      * Dark → Black + Purple secondary
    - ALWAYS provide secondary color when primary is determined

    SHAPE INFERENCE (from physical description):
    - Winged: "wings", "bird-like", "bat-like", "fly/flying", "soar"
    - Bipedal: "stands upright", "walks on two legs", "humanoid", "arms", "hands"
    - Quadruped: "four legs", "cat-like", "dog-like", "prowl", "wolf", "tiger"
    - Serpentine: "snake-like", "slithering", "long body", "coils"
    - Aquatic: "fish-like", "fins", "tail for swimming", "aquatic"
    - Insectoid: "bug-like", "antenna", "six legs", "exoskeleton"
    - Amorphous: "ghostly", "shapeless", "blob", "gaseous", "wispy"
    - Humanoid: "resembles a person", "martial artist", "human-like proportions"

    SIZE INFERENCE (from descriptors):
    - Tiny: "tiny", "miniature" (< 0.5m)
    - Small: "small", "little", "cute", "compact" (0.5-1.0m)
    - Medium: "moderate", default for most (1.0-2.0m)
    - Large: "large", "big", "imposing", "powerful build" (2.0-5.0m)
    - Huge: "huge", "massive", "gigantic", "towering" (> 5.0m)

    EVOLUTION INFERENCE:
    - Basic: "baby", "starter", "first form", "evolves from", default for small/cute Pokemon
    - Middle: "second form", "evolved", "intermediate stage"
    - Final: "final form", "fully evolved", "ultimate", large/powerful Pokemon
    - Single: "doesn't evolve", "standalone", legendaries

    HABITAT INFERENCE (from context clues):
    - Urban: "city", "urban environments", "buildings", "streets"
    - Mountain: "mountains", "peaks", "rocky", "high altitude", "volcanoes"
    - Sea: "ocean", "deep water", "seas", "aquatic environments"
    - Forest: "woods", "jungle", "trees", "woodland"
    - Grassland: "plains", "meadows", "fields", "savanna"
    - Cave: "underground", "caves", "caverns", "dark places"
    - Desert: "desert", "sand", "arid", "hot environments"
    - Tundra: "ice", "snow", "frozen", "arctic", "cold"
    - Wetland: "marsh", "swamp", "pond", "lake", "river"
    - Rare: legendaries, special environments

    LEGENDARY/MYTHICAL (be conservative):
    - Legendary: explicitly says "legendary" OR "extremely rare" + "powerful" + unique features
    - Mythical: explicitly says "mythical" OR "ancient" + "mysterious" + supernatural
    - Most Pokemon are NOT legendary/mythical

    {{_.role("user")}}
    Parse this Pokemon prompt into structured attributes:

    Prompt: "{{ user_prompt }}"

    Provide complete structured attributes. Remember: if the prompt is vague, infer reasonable Pokemon-like attributes.

    {{ ctx.output_format }}
  "#
}

// ----------------------------------------------------------------------------
// Test Cases
// ----------------------------------------------------------------------------

test TestSimplePrompt {
  functions [ParsePokemonPrompt]
  args {
    user_prompt "fire pokemon"
  }
}

test TestDetailedPrompt {
  functions [ParsePokemonPrompt]
  args {
    user_prompt "Create a water and dragon-type Pokemon with blue scales, elegant fins, and the ability to swim in deep oceans"
  }
}

test TestVaguePrompt {
  functions [ParsePokemonPrompt]
  args {
    user_prompt "cute yellow electric mouse"
  }
}

test TestLegendaryPrompt {
  functions [ParsePokemonPrompt]
  args {
    user_prompt "legendary fire bird with massive wings that soars above volcanoes"
  }
}
