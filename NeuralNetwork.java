import java.awt.image.*;
import java.io.*;
import java.nio.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.zip.*;

class NeuralNetwork {
  private int layers;
  private List<Matrix> bs;
  private List<Matrix> ws;

  public NeuralNetwork(int... sizes) {
    this.layers = sizes.length;
    this.bs = new ArrayList<>(sizes.length - 1);
    this.ws = new ArrayList<>(sizes.length - 1);

    for (int i = 1; i < sizes.length; ++i) {
      int neurons = sizes[i];
      int incomingNeurons = sizes[i-1];

      bs.add(new Matrix(neurons, 1).set(Function::random));
      ws.add(new Matrix(neurons, incomingNeurons).set(Function::random));
    }
  }

  public void train(List<Tuple<Matrix>> trainingData,
                    int minBatchSize,
                    double learningRate)
  {
    minBatchSize = Math.min(minBatchSize, trainingData.size());
    for (int k = 0; k < trainingData.size(); k += minBatchSize) {

      sgd(trainingData.subList(k, Math.min(trainingData.size(), k + minBatchSize)),
          learningRate);
    }
  }

  public Matrix predict(Matrix x) {
    return feedforward(x);
  }

  private void sgd(List<Tuple<Matrix>> trainingData, double learningRate) {
    List<Matrix> nablaB = zeros(bs);
    List<Matrix> nablaW = zeros(ws);

    for (Tuple<Matrix> xy : trainingData) {
      Tuple<List<Matrix>> deltaNablaBW = backprop(xy);

      zip(nablaB, deltaNablaBW.x, (nb, dnb) -> nb.incrementBy(dnb));
      zip(nablaW, deltaNablaBW.y, (nw, dnw) -> nw.incrementBy(dnw));
    }

    zip(ws, nablaW, (w, nw) -> w.decrementBy(nw.multiplyBy(learningRate / trainingData.size())));
    zip(bs, nablaB, (b, nb) -> b.decrementBy(nb.multiplyBy(learningRate / trainingData.size())));
  }

  private Matrix feedforward(Matrix x) {
    Matrix[] a = { x };

    zip(ws, bs, (w, b) -> {
        a[0] = w.dot(a[0]).incrementBy(b).set(Function::sigmoid);
      });

    return a[0];
  }

  private Tuple<List<Matrix>> backprop(Tuple<Matrix> xy) {
    List<Matrix> nablaB = new ArrayList<>(bs.size());
    List<Matrix> nablaW = new ArrayList<>(ws.size());

    List<Matrix> as = new ArrayList<>(layers);
    as.add(xy.x);

    List<Matrix> zs = new ArrayList<>(layers);

    zip(bs, ws, (b, w) -> {
        zs.add(w.dot(get(as, -1)).incrementBy(b));
        as.add(new Matrix(get(zs, -1)).set(Function::sigmoid));
      });

    Matrix delta = new Matrix(xy.y.getRow(), xy.y.getColumn());
    delta.set((i, j) -> ((get(as, -1).get(i, j)
                          - xy.y.get(i, j))
                         * Function.dSigmoid(get(zs, -1).get(i, j))));
    nablaB.add(delta);
    nablaW.add(delta.dot(get(as, -2).transpose()));

    for (int l = 2; l < layers; ++l) {
      Matrix z = get(zs, -l);
      Matrix sp = new Matrix(z).set(Function::dSigmoid);

      delta = sp.multiplyBy(get(ws, -l+1)
                            .transpose()
                            .dot(delta)
                            .get(0, 0));

      nablaB.add(delta);
      nablaW.add(delta.dot(get(as, -l-1).transpose()));
    }

    Collections.reverse(nablaB);
    Collections.reverse(nablaW);

    return new Tuple<>(nablaB, nablaW);
  }

  private <T> void zip(List<T> a, List<T> b, BiConsumer<T, T> consumer) {
    for (int i = 0; i < a.size(); ++i) {
      consumer.accept(a.get(i), b.get(i));
    }
  }

  private List<Matrix> zeros(List<Matrix> x) {
    List<Matrix> y = new ArrayList<>(x.size());

    for (Matrix i : x) {
      y.add(new Matrix(i.getRow(), i.getColumn()));
    }

    return y;
  }

  private Matrix get(List<Matrix> ms, int index) {
    return (0 <= index
            ? ms.get(index)
            : ms.get(ms.size() + index));

  }

  // ------------------------------------------------------------

  public static class Tuple<T> {
    public T x;
    public T y;

    public Tuple(T x, T y) {
      this.x = x;
      this.y = y;
    }
  }

  // ------------------------------------------------------------

  public interface Function {
    public double apply(double v);

    public static double random(double z) {
      return new java.util.Random().nextGaussian();
    }

    public static double sigmoid(double z) {
      return 1.0 / (1.0 + Math.exp(-z));
    }

    public static double dSigmoid(double z) {
      return sigmoid(z) * (1 - sigmoid(z));
    }
  }

  // ------------------------------------------------------------

  public interface MatrixFunction {
    public double apply(int i, int j);
  }

  public static class Matrix {
    private double[][] val;

    public Matrix(int row, int col) {
      val = new double[row][col];
    }

    public Matrix(double[][] val) {
      this.val = new double[val.length][0 < val.length ? val[0].length : 0];
      for (int i = 0; i < getRow(); ++i) {
        System.arraycopy(val[i], 0, this.val[i], 0, getColumn());
      }
    }

    public Matrix(Matrix m) {
      this(m.val);
    }

    public Matrix transpose() {
      return new Matrix(getColumn(), getRow())
        .set((i, j) -> get(j, i));
    }

    public Matrix incrementBy(Matrix b) {
      return set((i, j) -> get(i, j) + b.get(i, j));
    }

    public Matrix decrementBy(Matrix b) {
      return set((i, j) -> get(i, j) - b.get(i, j));
    }

    public Matrix multiplyBy(double v) {
      return set((i, j) -> get(i, j) * v);
    }

    public Matrix add(Matrix b) {
      Matrix a = this;

      return new Matrix(a.getRow(), a.getColumn())
        .set((i, j) -> a.get(i, j) + b.get(i, j));
    }

    public Matrix subtract(Matrix b) {
      Matrix a = this;

      return new Matrix(a.getRow(), a.getColumn())
        .set((i, j) -> a.get(i, j) - b.get(i, j));
    }

    public Matrix multiply(double v) {
      return new Matrix(getRow(), getColumn())
        .set((i, j) -> get(i, j) * v);
    }

    public Matrix dot(Matrix b) {
      if (getColumn() != b.getRow()) {
        throw new RuntimeException("Invalid dimension");
      }

      Matrix a = this;

      return new Matrix(a.getRow(), b.getColumn())
        .set((i, j) -> {
            double result = 0;
            for (int x = 0; x < this.getColumn(); ++x) {
              result += a.val[i][x] * b.val[x][j];
            }

            return result;
        });
    }

    public int getRow() { return val.length; }
    public int getColumn() { return 0 < getRow() ? val[0].length : 0; }

    public Matrix set(int i, int j, double value) {
      val[i][j] = value;
      return this;
    }

    public Matrix set(MatrixFunction f) {
      for (int i = 0; i < val.length; ++i) {
        for (int j = 0; j < val[i].length; ++j) {
          val[i][j] = f.apply(i, j);
        }
      }

      return this;
    }

    public Matrix set(Function f) {
      for (int i = 0; i < val.length; ++i) {
        for (int j = 0; j < val[i].length; ++j) {
          val[i][j] = f.apply(val[i][j]);
        }
      }

      return this;
    }

    public double get(int i, int j) { return val[i][j]; }

    public String toString() {
      StringBuilder buf = new StringBuilder();

      buf.append('[');

      for (int i = 0; i < getRow(); ++i) {
        if (0 < i) {
          buf.append(",\n ");
        }

        buf.append('[');
        for (int j = 0; j < getColumn(); ++j) {
          if (0 < j) buf.append(", ");
          buf.append(String.format("%5.3f", val[i][j]));
        }
        buf.append("]");
      }

      buf.append("]");

      return buf.toString();
    }
  }

  // ------------------------------------------------------------

  public static void main(String[] args)
    throws Exception
  {
    System.out.println(new Matrix(new double[][] { {1, 2, 3},
                                                   {4, 5, 6}})
      .incrementBy(new Matrix(new double[][] {{ 1, 1, 1 },
                                              { 2, 2, 2 }})));
    String dataDir = "/Users/quangkevin/Programming/Project/test/data";

    List<Tuple<Matrix>> training = neuralTuples(readInput(new File(dataDir + "/train-images-idx3-ubyte.gz")),
                                                readLabels(new File(dataDir + "/train-labels-idx1-ubyte.gz")));

    List<Tuple<Matrix>> testing = neuralTuples(readInput(new File(dataDir + "/t10k-images-idx3-ubyte.gz")),
                                               readLabels(new File(dataDir + "/t10k-labels-idx1-ubyte.gz")));

    NeuralNetwork network = new NeuralNetwork(training.get(0).x.getRow(),
                                              100,
                                              training.get(0).y.getRow());

    for (int epoch = 0; epoch < 30; ++epoch) {
      Collections.shuffle(training);
      network.train(training, 10, .5);

      int totalGood = 0;

      for (Tuple<Matrix> test : testing) {
        if (fromNeuralOutput(test.y) == fromNeuralOutput(network.predict(test.x))) {
          ++totalGood;
        }
      }

      System.out.printf("%s pred: %s actual: %s percentage: %s\n",
                        epoch,
                        totalGood,
                        testing.size(),
                        ((double) totalGood/testing.size()) * 100.0);
    }
  }

  private static List<Tuple<Matrix>> neuralTuples(Matrix[] x, int[] y) {
    List<Tuple<Matrix>> result = new ArrayList<>(x.length);

    for (int i = 0; i < x.length; ++i) {
      result.add(new Tuple<>(asNeuralInput(x[i]), asNeuralOutput(y[i])));
    }

    return result;
  }

  private static Matrix asNeuralInput(Matrix x) {
    Matrix result = new Matrix(x.getRow() * x.getColumn(), 1);
    for (int i = 0; i < result.getRow(); ++i) {
      result.set(i, 0, x.get(i / x.getColumn(),
                             i % x.getColumn()));
    }

    return result;
  }

  private static Matrix asNeuralOutput(int y) {
    Matrix result = new Matrix(10, 1);
    result.set(y, 0, 1.0);

    return result;
  }

  private static int fromNeuralOutput(Matrix output) {
    int max = 0;
    double maxVal = 0;

    for (int i = 0; i < output.getRow(); ++i) {
      if (maxVal < output.get(i, 0)) {
        max = i;
        maxVal = output.get(i, 0);
      }
    }

    return max;
  }

  private static Matrix[] readInput(File inputFile)
    throws Exception
  {
    try (InputStream in = new GZIPInputStream(new FileInputStream(inputFile))) {
      byte[] buf = new byte[4];
      readFull(in, buf, 4); // magic number

      readFull(in, buf, 4); // length
      int inputSize = asInt(buf);

      readFull(in, buf, 4);
      int numRows = asInt(buf);

      readFull(in, buf, 4);
      int numCols = asInt(buf);

      buf = new byte[inputSize * numRows * numCols];
      readFull(in, buf, buf.length);

      Matrix[] input = new Matrix[inputSize];
      int count = 0;
      for (int i = 0; i < inputSize; ++i) {
        input[i] = new Matrix(numRows, numCols);

        for (int j = 0; j < numRows; ++j) {
          for (int k = 0; k < numCols; ++k) {
            input[i].set(j, k, ((double) (buf[count++] & 0xFF)) / 255.0);
          }
        }
      }

      return input;
    }
  }

  private static int[] readLabels(File labelFile)
    throws Exception
  {
    try (InputStream in = new GZIPInputStream(new FileInputStream(labelFile))) {
      byte[] buf = new byte[4];
      readFull(in, buf, 4); // magic number;

      readFull(in, buf, 4); // length
      int length = asInt(buf);

      buf = new byte[length];
      readFull(in, buf, length);

      int[] labels = new int[length];
      for (int i = 0; i < labels.length; ++i) {
        labels[i] = buf[i] & 0xFF;
      }

      return labels;
    }
  }

  private static void readFull(InputStream in, byte[] buf, int total)
    throws Exception
  {
    int off = 0;
    while (off < total) {
      off += in.read(buf, off, total - off);
    }
  }

  private static int asInt(byte[] bytesInBigEndian) {
    return ByteBuffer.wrap(bytesInBigEndian).order(ByteOrder.BIG_ENDIAN).getInt();
  }

  private static int f(int i) {
    return i * i + 1;
  }

  private static Matrix toMatrix(int val) {
    return new Matrix(32, 1)
      .set((i, j) -> 0 == ((val >>> i) & 1) ? 0 : 1);
  }

  private static int fromMatrix(Matrix x) {
    int val = 0;
    for (int i = 0; i < x.getRow(); ++i) {
      val += (1 == x.get(i, 0) ? 1 << i : 0);
    }

    return val;
  }

  private static void generateImage(Matrix m)
    throws Exception
  {
    BufferedImage image = new BufferedImage(m.getColumn(), m.getRow(), BufferedImage.TYPE_BYTE_GRAY);
    for (int i = 0; i < m.getRow(); ++i) {
      for (int j = 0; j < m.getColumn(); ++j) {
        image.setRGB(j, i, ((int) m.get(i, j)) * 0x00010101);
      }
    }

    javax.imageio.ImageIO.write(image, "jpg", new File("/Users/quangkevin/Downloads/test.jpg"));

  }
}
