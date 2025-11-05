import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError, Subject } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';

export interface GenerateRequest {
  prompt: string;
  num_samples?: number;
  use_llm_parsing?: boolean;
}

export interface GenerateResponse {
  success: boolean;
  image: string;
  prompt: string;
  message?: string;
}

export interface ProgressiveImage {
  step: number;
  total_steps: number;
  image: string;
  is_final: boolean;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device?: string;
  message?: string;
}

@Injectable({
  providedIn: 'root',
})
export class PokemonApiService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = 'http://localhost:8000/api/v1';

  generatePokemon(
    prompt: string,
    numSamples: number = 1,
    useLlmParsing: boolean = true
  ): Observable<GenerateResponse> {
    const request: GenerateRequest = {
      prompt,
      num_samples: numSamples,
      use_llm_parsing: useLlmParsing,
    };

    return this.http
      .post<GenerateResponse>(`${this.apiUrl}/generate`, request)
      .pipe(retry(1), catchError(this.handleError));
  }

  checkHealth(): Observable<HealthResponse> {
    return this.http
      .get<HealthResponse>(`${this.apiUrl}/health`)
      .pipe(catchError(this.handleError));
  }

  generatePokemonStream(prompt: string, numSteps: number = 5): Observable<ProgressiveImage> {
    const subject = new Subject<ProgressiveImage>();

    const url = `${this.apiUrl}/generate/stream?prompt=${encodeURIComponent(
      prompt
    )}&num_steps=${numSteps}`;

    const eventSource = new EventSource(url);

    eventSource.onmessage = (event) => {
      try {
        const data: ProgressiveImage = JSON.parse(event.data);
        subject.next(data);

        if (data.is_final) {
          eventSource.close();
          subject.complete();
        }
      } catch (error) {
        subject.error(new Error('Failed to parse stream data'));
        eventSource.close();
      }
    };

    eventSource.onerror = (error) => {
      subject.error(new Error('Stream connection failed'));
      eventSource.close();
    };

    return subject.asObservable();
  }

  private handleError(error: HttpErrorResponse): Observable<never> {
    let errorMessage = 'An unknown error occurred';

    if (error.error instanceof ErrorEvent) {
      // Client-side error
      errorMessage = `Error: ${error.error.message}`;
    } else {
      // Server-side error
      errorMessage = `Error Code: ${error.status}\nMessage: ${error.message}`;
      if (error.error?.detail) {
        errorMessage = error.error.detail;
      }
    }

    return throwError(() => new Error(errorMessage));
  }
}
