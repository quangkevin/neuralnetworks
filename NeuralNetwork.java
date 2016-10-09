import java.awt.image.*;
import java.io.*;
import java.nio.*;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.zip.*;

class NeuralNetwork {
  private int layers;
  private List<Matrix> bs;
  private List<Matrix> ws;
  private Cost cost = new CrossEntropyCost();

  public NeuralNetwork(int... sizes) {
    this.layers = sizes.length;
    this.bs = new ArrayList<>(sizes.length - 1);
    this.ws = new ArrayList<>(sizes.length - 1);

    Random random = new Random();

    for (int i = 1; i < sizes.length; ++i) {
      int neurons = sizes[i];
      int incomingNeurons = sizes[i-1];

      bs.add(new Matrix(neurons, 1).set((x, y) -> random.nextGaussian()));
      ws.add(new Matrix(neurons, incomingNeurons).set((x, y) -> random.nextGaussian()));
    }
  }

  public NeuralNetwork(File f)
    throws Exception
  {
    restore(f);
  }

  public void train(List<Tuple<Matrix>> training,
                    int epochs,
                    int minBatchSize,
                    double learningRate,
                    double regularizationParameter,
                    java.util.function.Consumer<Integer> consumer)
  {
    minBatchSize = Math.min(minBatchSize, training.size());

    for (int epoch = 0; epoch < epochs; ++epoch) {
      Collections.shuffle(training);

      for (int k = 0; k < training.size(); k += minBatchSize) {
        sgd(training.subList(k, Math.min(training.size(), k + minBatchSize)),
            learningRate,
            regularizationParameter,
            training.size());
      }

      if (null != consumer) {
        consumer.accept(epoch);
      }
    }
  }

  public Matrix predict(Matrix x) {
    return feedforward(x);
  }

  public void save(File f)
    throws IOException
  {
    try (DataOutputStream out = new DataOutputStream(new FileOutputStream(f))) {
      out.writeInt(layers);
      for (Matrix b : bs) {
        out.writeInt(b.getRow());
        out.writeInt(b.getColumn());
        for (int i = 0; i < b.getRow(); ++i) {
          for (int j = 0; j < b.getColumn(); ++j) {
            out.writeDouble(b.get(i, j));
          }
        }
      }
      for (Matrix w : ws) {
        out.writeInt(w.getRow());
        out.writeInt(w.getColumn());
        for (int i = 0; i < w.getRow(); ++i) {
          for (int j = 0; j < w.getColumn(); ++j) {
            out.writeDouble(w.get(i, j));
          }
        }
      }
    }
  }

  public void restore(File f)
    throws Exception
  {
    try (DataInputStream in = new DataInputStream(new FileInputStream(f))) {
      this.layers = in.readInt();

      this.bs = new ArrayList<>(this.layers - 1);
      for (int i = 0; i < (this.layers - 1); ++i) {
        Matrix b = new Matrix(in.readInt(), in.readInt());
        for (int j = 0; j < b.getRow(); ++j) {
          for (int k = 0; k < b.getColumn(); ++k) {
            b.set(j, k, in.readDouble());
          }
        }
        bs.add(b);
      }

      this.ws = new ArrayList<>(this.layers - 1);
      for (int i = 0; i < (this.layers - 1); ++i) {
        Matrix w = new Matrix(in.readInt(), in.readInt());
        for (int j = 0; j < w.getRow(); ++j) {
          for (int k = 0; k < w.getColumn(); ++k) {
            w.set(j, k, in.readDouble());
          }
        }

        ws.add(w);
      }
    }
  }

  private void sgd(List<Tuple<Matrix>> training,
                   double learningRate,
                   double regularizationParameter,
                   int totalTrainingSize) {
    List<Matrix> nablaB = zeros(bs);
    List<Matrix> nablaW = zeros(ws);

    for (Tuple<Matrix> xy : training) {
      Tuple<List<Matrix>> deltaNablaBW = backprop(xy);

      zip(nablaB, deltaNablaBW.x, (nb, dnb) -> nb.incrementBy(dnb));
      zip(nablaW, deltaNablaBW.y, (nw, dnw) -> nw.incrementBy(dnw));
    }

    zip(ws, nablaW, (w, nw) -> w
        .multiplyBy(1.0 - learningRate * (regularizationParameter/totalTrainingSize))
        .decrementBy(nw.multiplyBy(learningRate / training.size())));

    zip(bs, nablaB, (b, nb) -> b.decrementBy(nb.multiplyBy(learningRate / training.size())));
  }

  private Matrix feedforward(Matrix x) {
    Matrix a = x;

    for (int i = 0; i < ws.size(); ++i) {
      Matrix w = ws.get(i);
      Matrix b = bs.get(i);

      a = w.dot(a).incrementBy(b).set(Function::sigmoid);
    }

    return a;
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

    Matrix delta = cost.delta(get(zs, -1), get(as, -1), xy.y);
    nablaB.add(delta);
    nablaW.add(delta.dot(get(as, -2).transpose()));

    for (int l = 2; l < layers; ++l) {
      Matrix z = get(zs, -l);
      Matrix sp = new Matrix(z).set(Function::dsigmoid);

      delta = sp.multiplyBy(get(ws, -l+1).transpose().dot(delta));

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

  private Matrix normalize(List<Tuple<Matrix>> trainingSamples) {
    Matrix normalizationFactor = new Matrix(trainingSamples.get(0).x.getRow(), 1);
    for (Tuple<Matrix> m : trainingSamples) {
      normalizationFactor.set((i, j) -> normalizationFactor.get(i,j) + m.x.get(i, j));
    }

    normalizationFactor.set((i, j) -> normalizationFactor.get(i, j) / trainingSamples.size());

    for (Tuple<Matrix> m : trainingSamples) {
      m.x.decrementBy(normalizationFactor);
    }

    return normalizationFactor;
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

  public interface Cost {
    public Matrix delta(Matrix z, Matrix a, Matrix y);
    public double cost(Matrix a, Matrix y);
  }

  // ------------------------------------------------------------

  public static class QuadraticCost implements Cost {
    public double cost(Matrix a, Matrix y) {
      double total = 0;
      for (int i = 0; i < a.getRow(); ++i) {
        for (int j = 0; j < a.getColumn(); ++i) {
          total += Math.pow(a.get(i, j) - y.get(i, j), 2);
        }
      }

      return total / 2;
    }

    public Matrix delta(Matrix z, Matrix a, Matrix y) {
      return a.subtract(y).set(Function::dsigmoid);
    }
  }

  // ------------------------------------------------------------

  public static class CrossEntropyCost implements Cost {
    public double cost(Matrix a, Matrix y) {
      double total = 0;
      for (int i = 0; i < a.getRow(); ++i) {
        for (int j = 0; j < a.getColumn(); ++j) {
          double val = (-y.get(i, j) * Math.log(a.get(i, j))
                        - (1 - y.get(i, j)) * Math.log(1 - a.get(i, j)));
          if (!Double.isNaN(val)) {
            total += val;
          }
        }
      }

      return total;
    }

    public Matrix delta(Matrix z, Matrix a, Matrix y) {
      return a.subtract(y);
    }
  }

  // ------------------------------------------------------------

  public interface Callback {
    public void invoke();
  }

  public interface Function {
    public double apply(double v);

    public static double normal(double z) {
      return new java.util.Random().nextGaussian() * z;
    }

    public static double sigmoid(double z) {
      return 1.0 / (1.0 + Math.exp(-z));
    }

    public static double dsigmoid(double z) {
      return sigmoid(z) * (1 - sigmoid(z));
    }

    public static double tanh(double z) {
      return Math.tanh(z);
    }

    public static double dtanh(double z) {
      double y = Math.tanh(z);

      return (1 - y * y);
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

    public Matrix multiplyBy(Matrix b) {
      return set((i, j) -> get(i, j) * b.get(i, j));
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
        throw new RuntimeException(String.format("Cannot dot %sx%s with %sx%s",
                                                 getRow(), getColumn(),
                                                 b.getRow(), b.getColumn()));
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

    public double scalar() {
      if (1 != getRow() || 1 != getColumn()) {
        throw new IllegalStateException("Not valid for "
                                        + getRow() + "x" + getColumn()
                                        + " matrix.");
      }

      return get(0, 0);
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
          //buf.append(String.format("%5.3f", val[i][j]));
          buf.append(val[i][j]);
        }
        buf.append("]");
      }

      buf.append("]");

      return buf.toString();
    }
  }

  // ------------------------------------------------------------

  public static void test()
    throws Exception
  {
    List<Tuple<Matrix>> training = new ArrayList<>();
    for (int i = 0; i < 1; ++i) {
      training.add(new Tuple<>(new Matrix(new double[][] {{ 0 },
                                                          { 0 }}),
          new Matrix(new double[][] {{ 0 }})));
      training.add(new Tuple<>(new Matrix(new double[][] {{ 0 },
                                                          { 1 }}),
          new Matrix(new double[][] {{ 1 }})));
      training.add(new Tuple<>(new Matrix(new double[][] {{ 1 },
                                                          { 0 }}),
          new Matrix(new double[][] {{ 1 }})));
      training.add(new Tuple<>(new Matrix(new double[][] {{ 1 },
                                                          { 1 }}),
          new Matrix(new double[][] {{ 0 }})));
    }

    NeuralNetwork network = new NeuralNetwork(2, 3, 1);
    network.train(new ArrayList<>(training), 1000, 1, .5, 0, x -> {});

    for (Tuple<Matrix> t : training) {
      System.out.println(t.y);
      System.out.println("==>");
      System.out.println(network.predict(t.x));
      System.out.println("-----");
    }
  }

  public static void main(String[] args)
    throws Exception
  {
    //test();
    //if (true) return;
    String dataDir = "data";

    List<Tuple<Matrix>> training = neuralTuples(readInput(new File(dataDir + "/train-images-idx3-ubyte.gz")),
                                                readLabels(new File(dataDir + "/train-labels-idx1-ubyte.gz")));

    List<Tuple<Matrix>> testing = neuralTuples(readInput(new File(dataDir + "/t10k-images-idx3-ubyte.gz")),
                                               readLabels(new File(dataDir + "/t10k-labels-idx1-ubyte.gz")));

    NeuralNetwork network = new NeuralNetwork(training.get(0).x.getRow(),
                                              30,
                                              training.get(0).y.getRow());

    network.train(training, 100, 10, .5, 1, epoch -> {
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
      });
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

    javax.imageio.ImageIO.write(image, "jpg", new File("/tmp/test.jpg"));

  }
}
