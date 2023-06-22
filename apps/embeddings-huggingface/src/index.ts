import { HfInference } from '@huggingface/inference';
import { config } from 'dotenv';

config({ path: '.env.local' });

const hf = new HfInference(process.env.HF_TOKEN);

function dotProduct(a: number[], b: number[]) {
  if (a.length !== b.length) {
    throw new Error('Both arguments must have the same length');
  }

  let result = 0;

  for (let i = 0; i < a.length; i++) {
    result += a[i] * b[i];
  }

  return result;
}

const output1 = await hf.featureExtraction({
  model: 'sentence-transformers/all-MiniLM-L6-v2',
  inputs: 'That is a happy person',
});

const output2 = await hf.featureExtraction({
  model: 'sentence-transformers/all-MiniLM-L6-v2',
  inputs: 'That is a happy person',
});

if (is1DArray(output1) && is1DArray(output2)) {
  const similarity = dotProduct(output1, output2);

  console.log(similarity);
}

function is1DArray<T>(value: (T | T[] | T[][])[]): value is T[] {
  return !Array.isArray(value[0]);
}
