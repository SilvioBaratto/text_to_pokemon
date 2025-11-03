import { Component, signal, viewChild, inject } from '@angular/core';
import { ChatInputComponent } from './components/chat-input/chat-input.component';
import { PokemonDisplayComponent, PokemonGeneration } from './components/pokemon-display/pokemon-display.component';
import { PokemonApiService } from './services/pokemon-api.service';

@Component({
  selector: 'app-root',
  imports: [ChatInputComponent, PokemonDisplayComponent],
  templateUrl: './app.html',
  standalone: true
})
export class App {
  private readonly pokemonApi = inject(PokemonApiService);

  chatInput = viewChild<ChatInputComponent>('chatInput');

  currentPokemon = signal<PokemonGeneration | null>(null);
  isLoading = signal(false);
  error = signal<string | null>(null);

  onPromptSubmit(prompt: string): void {
    this.isLoading.set(true);
    this.error.set(null);

    const chatInputComponent = this.chatInput();
    if (chatInputComponent) {
      chatInputComponent.setDisabled(true);
    }

    // Use simple POST /generate endpoint
    this.pokemonApi.generatePokemon(prompt, 1).subscribe({
      next: (response) => {
        this.isLoading.set(false);

        // Update display with generated Pokemon
        this.currentPokemon.set({
          image: response.image,
          prompt: response.prompt,
          timestamp: new Date(),
          isProgressive: false
        });

        // Clear input after successful generation
        if (chatInputComponent) {
          chatInputComponent.clear();
          chatInputComponent.setDisabled(false);
        }
      },
      error: (err) => {
        this.error.set(err.message || 'Failed to generate Pokemon. Please ensure the API server is running.');
        this.isLoading.set(false);
        if (chatInputComponent) {
          chatInputComponent.setDisabled(false);
        }
      }
    });
  }
}
