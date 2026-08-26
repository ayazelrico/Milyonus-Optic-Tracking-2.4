export class ParameterEngine {
  private params: Float32Array;

  constructor(size: number = 300000) {
    this.params = new Float32Array(size);
    // Initialize with small random values to simulate a pre-trained state
    for (let i = 0; i < size; i++) {
      this.params[i] = (Math.random() - 0.5) * 0.1;
    }
  }

  /**
   * Modulates input values using a slice of parameters.
   * Uses a weighted sum followed by a sigmoid activation.
   */
  modulate(inputs: number[], offset: number): number {
    if (offset + inputs.length > this.params.length) {
      throw new Error("Parameter offset exceeds engine bounds");
    }

    let sum = 0;
    for (let i = 0; i < inputs.length; i++) {
      sum += inputs[i] * this.params[offset + i];
    }

    // Sigmoid activation to keep result in [0, 1]
    return 1 / (1 + Math.exp(-sum));
  }
}
