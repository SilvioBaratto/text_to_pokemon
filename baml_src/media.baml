enum PokemonType {
  Normal
  Fire
  Water
  Electric
  Grass
  Ice
  Fighting
  Poison
  Ground
  Flying
  Psychic
  Bug
  Rock
  Ghost
  Dragon
  Dark
  Steel
  Fairy
}

enum PokemonColor {
  Red
  Blue
  Yellow
  Green
  Purple
  Orange
  Pink
  Brown
  Black
  White
  Gray
}

enum BodyShape {
  Bipedal      // Stands on two legs (e.g., Charizard, Blaziken)
  Quadruped    // Walks on four legs (e.g., Pikachu, Eevee)
  Serpentine   // Snake-like body (e.g., Dragonair, Onix)
  Winged       // Primarily flies (e.g., Pidgeot, Articuno)
  Amorphous    // No fixed form (e.g., Ditto, Gastly)
  Insectoid    // Insect-like (e.g., Beedrill, Scizor)
  Aquatic      // Fish/water creature (e.g., Magikarp, Kyogre)
  Humanoid     // Human-like proportions (e.g., Machamp, Gardevoir)
}

enum SizeClass {
  Tiny         // < 0.5m (e.g., Joltik, Flabebe)
  Small        // 0.5m - 1.0m (e.g., Pikachu, Eevee)
  Medium       // 1.0m - 2.0m (e.g., Charizard, Lucario)
  Large        // 2.0m - 5.0m (e.g., Onix, Steelix)
  Huge         // > 5.0m (e.g., Wailord, Eternatus)
}

enum Habitat {
  Forest       // Woods, jungles
  Grassland    // Plains, meadows
  Mountain     // Rocky areas, caves
  Cave         // Underground
  Sea          // Ocean, deep water
  Urban        // Cities, buildings
  Rare         // Special/legendary locations
  Desert       // Sandy, arid areas
  Tundra       // Snowy, icy regions
  Wetland      // Marshes, swamps
}

enum EvolutionStage {
  Basic        // First form (e.g., Charmander)
  Middle       // Second form (e.g., Charmeleon)
  Final        // Third/final form (e.g., Charizard)
  Single       // Doesn't evolve (e.g., Tauros, Absol)
  Mega         // Mega evolution
  Gigantamax   // Gigantamax form
}

// ----------------------------------------------------------------------------
// Classes - Structured Metadata
// ----------------------------------------------------------------------------

class ColorDistribution {
  primary PokemonColor      // Most dominant color
  secondary PokemonColor?   // Second most visible color (optional)
  accent PokemonColor?      // Small accent/detail color (optional)
}

class VisualAttributes {
  primary_colors PokemonColor[]       // All primary colors (list)
  secondary_colors PokemonColor[]     // All secondary/accent colors
  color_distribution ColorDistribution // Which color dominates
  body_shape BodyShape                // Overall body structure
  size_class SizeClass                // Relative size category
  distinctive_features string[]       // Visual features (e.g., "pointy_ears", "flames_on_tail")
}

class CategoricalAttributes {
  type1 PokemonType          // Primary type
  type2 PokemonType?         // Secondary type (optional)
  generation int             // Pokemon generation (1-9)
  is_legendary bool          // Is it a legendary Pokemon?
  is_mythical bool           // Is it a mythical Pokemon?
  is_starter bool            // Is it a starter Pokemon?
  evolution_stage EvolutionStage
}

class SemanticAttributes {
  personality_traits string[]  // e.g., ["aggressive", "protective", "playful"]
  habitat Habitat              // Natural environment
  rarity string                // "common", "uncommon", "rare", "legendary"
  combat_style string          // e.g., "physical_attacker", "special_defender", "tank"
}

class PokemonMetadata {
  // Text prompts for semantic conditioning (5-7 diverse descriptions)
  prompts string[]

  // Visual attributes
  visual_attributes VisualAttributes

  // Categorical attributes (for enum conditioning)
  categorical_attributes CategoricalAttributes

  // Semantic attributes (additional context)
  semantic_attributes SemanticAttributes
}

// ----------------------------------------------------------------------------
// BAML Function - Extract Pokemon Metadata from Image
// ----------------------------------------------------------------------------

function ExtractPokemonMetadata(img: image, pokemon_name: string) -> PokemonMetadata {
  client CustomGPT5Mini

  prompt #"
    {{_.role("system")}}
    You are a Pokemon expert extracting structured metadata from Pokemon images for training a Conditional VAE model. Your output will be used for ML training, so accuracy is critical.

    KEY RULES:
    - Prompts: Generate 5-7 user-style requests starting with "Create", "I want", "Generate", "Make", "Imagine", "Design", "Give me", or "Show me" (10-25 words each, diverse, natural language)
    - Colors: Identify all visible colors accurately (primary vs secondary/accent)
    - Types: Infer Pokemon type(s) from visual appearance
    - Features: List 3-5 distinctive visual traits
    - Attributes: Include generation, legendary/mythical/starter status, evolution stage, personality, habitat, rarity, combat style

    {{_.role("user")}}
    Analyze this Pokemon image (species: {{ pokemon_name }}) and extract:
    1. 5-7 diverse user-style prompts (e.g., "Create a fire-type Pokemon with orange scales and blue wings")
    2. Visual attributes: primary/secondary colors, color distribution, body shape, size class, distinctive features
    3. Categorical attributes: type1/type2, generation (1-9), legendary/mythical/starter flags, evolution stage
    4. Semantic attributes: personality traits (2-4), habitat, rarity, combat style

    Image: {{ img }}

    {{ ctx.output_format }}
  "#
}

// ----------------------------------------------------------------------------
// Test Cases
// ----------------------------------------------------------------------------

test TestMagearna {
  functions [ExtractPokemonMetadata]
  args {
    img {
      url "images/Magearna/0.png"
    }
    pokemon_name "Magearna"
  }
}

// Additional test with a different Pokemon type for validation
test TestPikachu {
  functions [ExtractPokemonMetadata]
  args {
    img {
      url "images/Pikachu/0.png"
    }
    pokemon_name "Pikachu"
  }
}
