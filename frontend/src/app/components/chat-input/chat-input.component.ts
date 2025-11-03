import { Component, signal, output } from '@angular/core';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-chat-input',
  imports: [FormsModule],
  templateUrl: './chat-input.component.html',
  standalone: true
})
export class ChatInputComponent {
  prompt = signal('');
  isDisabled = signal(false);

  sendPrompt = output<string>();

  onSubmit(): void {
    const text = this.prompt().trim();
    if (text && !this.isDisabled()) {
      this.sendPrompt.emit(text);
    }
  }

  onKeyPress(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.onSubmit();
    }
  }

  setDisabled(disabled: boolean): void {
    this.isDisabled.set(disabled);
  }

  clear(): void {
    this.prompt.set('');
  }
}
