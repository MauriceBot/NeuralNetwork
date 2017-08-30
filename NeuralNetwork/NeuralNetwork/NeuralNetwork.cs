using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        private int inputnodes;
        private int hiddennodes;
        private int outputnodes;
        private double learningrate;

        Matrix weightInputHidden = new Matrix(0, 0);
        Matrix weightHiddenOutput = new Matrix(0, 0);

        public NeuralNetwork(int inputnodes, int hiddennodes, int outputnodes, double learningrate)
        {
            this.inputnodes = inputnodes;
            this.hiddennodes = hiddennodes;
            this.outputnodes = outputnodes;
            this.learningrate = learningrate;
            weightInputHidden = RandomizeWeights(inputnodes, hiddennodes);
            weightHiddenOutput = RandomizeWeights(hiddennodes, outputnodes);
        }

        public Matrix Query(Matrix inputs)
        {
            // Inputs going into the hidden layer.
            Matrix hidden_inputs = Matrix.DotProduct(inputs, weightInputHidden);
            // Inputs leaving the hidden layer sigmoided.
            Matrix hidden_outputs = hidden_inputs.Sigmoid();

            // Inputs going into the final layer.
            Matrix final_inputs = Matrix.DotProduct(hidden_outputs, weightHiddenOutput);
            // Inputs leaving the final layer sigmoided.
            Matrix final_outputs = final_inputs.Sigmoid();

            return final_outputs;
        }

        private Matrix RandomizeWeights(int r, int c)
        {
            Random rand = new Random();
            Matrix temp = new Matrix(r, c);
            for (int i = 0; i < temp.GetLength(0); i++)
                for (int j = 0; j < temp.GetLength(1); j++)
                    temp[i, j] = rand.NextDouble() - 0.5;
            return temp;
        }

        public void train(Matrix inputs, Matrix target)
        {
            // Inputs going into the hidden layer.
            Matrix hidden_inputs = Matrix.DotProduct(inputs, weightInputHidden);
            // Inputs leaving the hidden layer sigmoided.
            Matrix hidden_outputs = hidden_inputs.Sigmoid();

            // Inputs going into the final layer.
            Matrix final_inputs = Matrix.DotProduct(hidden_outputs, weightHiddenOutput);
            // Inputs leaving the final layer sigmoided.
            Matrix final_outputs = final_inputs.Sigmoid();

            // Errors in the output layer.
            Matrix output_errors = target - final_outputs;
            // Errors in the hidden layer.
            Matrix hidden_errors = Matrix.DotProduct(output_errors, weightHiddenOutput.Transpose());

            UpdateWeights(weightHiddenOutput, output_errors, final_outputs, hidden_outputs);
            UpdateWeights(weightInputHidden, hidden_errors, hidden_outputs, inputs);
        }

        private void UpdateWeights(Matrix weights, Matrix errors, Matrix first_outputs, Matrix second_outputs)
        {
            Matrix temp5 = errors * first_outputs * (1.0 - first_outputs);
            Matrix temp2 = Matrix.DotProduct(second_outputs.Transpose(), temp5);
            for (int i = 0; i < weights.GetLength(0); i++)
                for (int j = 0; j < weights.GetLength(1); j++)
                    weights[i, j] += temp2[i, j] * learningrate;
        }

        static void Main(string[] args)
        {
            NeuralNetwork test = new NeuralNetwork(784, 100, 10, 0.3);
            using (var reader = new System.IO.StreamReader(@"mnist_train.csv"))
            {
                int label;
                while (!reader.EndOfStream)
                {

                    var line = reader.ReadLine();
                    var values = line.Split(',');
                    Matrix inputs = new Matrix(1, 784);

                    label = int.Parse(values[0]);
                    for (int i = 1; i < values.Length - 1; i++)
                    {
                        inputs[0, i] = (int.Parse(values[i]) / 255.0 * 0.99) + 0.01;
                    }
                    Matrix targets = new Matrix(1, 10);
                    for (int i = 0; i < targets.GetLength(1); i++)
                    {
                        targets[0, i] = 0.01;
                    }
                    targets[0, label] = 0.99;

                    test.train(inputs, targets);
                }
            }

            using (var reader = new System.IO.StreamReader(@"mnist_test.csv"))
            {
                int label;
                var line = reader.ReadLine();
                var values = line.Split(',');
                Matrix inputs = new Matrix(1, 784);

                label = int.Parse(values[0]);
                for (int i = 1; i < values.Length - 1; i++)
                {
                    inputs[0, i] = (int.Parse(values[i]) / 255.0 * 0.99) + 0.01;
                }

                Matrix trueresults = test.Query(inputs);
            }
        }
    }
}
