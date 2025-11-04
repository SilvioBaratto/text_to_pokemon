import { Component, signal, input, effect } from '@angular/core';
import { CommonModule, DecimalPipe, DatePipe } from '@angular/common';

export interface PokemonGeneration {
  image: string;
  prompt: string;
  timestamp: Date;
  isProgressive?: boolean;
  evolutionStep?: number;
  totalSteps?: number;
}

@Component({
  selector: 'app-pokemon-display',
  standalone: true,
  imports: [CommonModule, DecimalPipe, DatePipe],
  templateUrl: './pokemon-display.component.html'
})
export class PokemonDisplayComponent {
  currentPokemon = input<PokemonGeneration | null>(null);
  isLoading = input<boolean>(false);
  error = input<string | null>(null);

  // Animation state
  showAnimation = signal(false);

  // Progressive evolution state
  evolutionProgress = signal(0);
  showEvolutionEffect = signal(false);

  constructor() {
    // Trigger animation when new Pokemon is received
    effect(() => {
      const pokemon = this.currentPokemon();
      if (pokemon) {
        if (pokemon.isProgressive && pokemon.evolutionStep && pokemon.totalSteps) {
          // Progressive evolution: show progress
          this.evolutionProgress.set((pokemon.evolutionStep / pokemon.totalSteps) * 100);
          this.showEvolutionEffect.set(true);

          // Only trigger full animation on final step
          if (pokemon.evolutionStep === pokemon.totalSteps) {
            this.showAnimation.set(false);
            setTimeout(() => {
              this.showAnimation.set(true);
              this.showEvolutionEffect.set(false);
            }, 100);
          }
        } else {
          // Regular generation: full animation
          this.showAnimation.set(false);
          this.showEvolutionEffect.set(false);
          setTimeout(() => this.showAnimation.set(true), 50);
        }
      }
    });
  }

  downloadPokemon(): void {
    const pokemon = this.currentPokemon();
    if (!pokemon) return;

    const link = document.createElement('a');
    link.href = pokemon.image;
    link.download = `pokemon_${pokemon.timestamp.getTime()}.png`;
    link.click();
  }
}
