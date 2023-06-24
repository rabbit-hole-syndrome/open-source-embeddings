import { pipeline } from '@xenova/transformers';

const generateEmbeddings = await pipeline(
  'feature-extraction',
  'Xenova/all-MiniLM-L6-v2'
);

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

const output1 = await generateEmbeddings('That is a happy person', {
  pooling: 'mean',
  normalize: true,
});

const output2 = await generateEmbeddings('That is a sad person', {
  pooling: 'mean',
  normalize: true,
});

const similarity = dotProduct(output1.data, output2.data);

console.log(similarity);
