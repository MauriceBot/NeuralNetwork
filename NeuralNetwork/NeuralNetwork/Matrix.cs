using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Matrix
    {
        public const int DimSize = 3;
        private double[,] m_matrix = new double[DimSize, DimSize];

        public Matrix(int rows, int columns)
        {
            m_matrix = new double[rows, columns];
        }

        public static Matrix operator -(Matrix mat1, Matrix mat2)
        {
            int rows = mat1.m_matrix.GetLength(0);
            int columns = mat1.m_matrix.GetLength(1);
            Matrix newMatrix = new Matrix(rows, columns);

            for (int x = 0; x < rows; x++)
                for (int y = 0; y < columns; y++)
                    newMatrix[x, y] = mat1[x, y] - mat2[x, y];

            return newMatrix;
        }

        public static Matrix operator -(double number, Matrix mat1)
        {
            int rows = mat1.m_matrix.GetLength(0);
            int columns = mat1.m_matrix.GetLength(1);
            Matrix newMatrix = new Matrix(rows, columns);

            for (int x = 0; x < rows; x++)
                for (int y = 0; y < columns; y++)
                    newMatrix[x, y] = number - mat1[x, y];

            return newMatrix;
        }

        public static Matrix operator *(Matrix mat1, Matrix mat2)
        {
            int rows = mat1.m_matrix.GetLength(0);
            int columns = mat1.m_matrix.GetLength(1);
            Matrix newMatrix = new Matrix(rows, columns);

            for (int x = 0; x < rows; x++)
                for (int y = 0; y < columns; y++)
                    newMatrix[x, y] = mat1[x, y] * mat2[x, y];

            return newMatrix;
        }

        internal void TwoDimensions()
        {
            double[,] temp = new double[2, m_matrix.GetLength(1)];
            for (int i = 0; i < temp.GetLength(1); i++)
            {
                temp[0, i] = m_matrix[0, i];
                temp[1, i] = 1;
            }
            m_matrix = temp;

        }

        public static Matrix DotProduct(Matrix a, Matrix b) // Mutltithread this?
        {

            Matrix c = new Matrix(a.m_matrix.GetLength(0), b.m_matrix.GetLength(1));
            for (int i = 0; i < c.m_matrix.GetLength(0); i++)
            {
                for (int j = 0; j < c.m_matrix.GetLength(1); j++)
                {
                    c[i, j] = 0;
                    for (int k = 0; k < a.m_matrix.GetLength(1); k++) // OR k<b.GetLength(0)
                        c[i, j] = c[i, j] + a[i, k] * b[k, j];
                }
            }
            return c;
        }

        public Matrix Transpose()
        {
            int w = m_matrix.GetLength(0);
            int h = m_matrix.GetLength(1);

            Matrix result = new Matrix(h, w);

            for (int i = 0; i < w; i++)
                for (int j = 0; j < h; j++)
                    result[j, i] = m_matrix[i, j];

            return result;
        }

        public Matrix Sigmoid()
        {
            int w = m_matrix.GetLength(0);
            int h = m_matrix.GetLength(1);

            Matrix result = new Matrix(w, h);

            for (int i = 0; i < w; i++)
                for (int j = 0; j < h; j++)
                    result[i, j] = 1 / (1 + Math.Exp(-m_matrix[i, j])); ;

            return result;
        }

        public int GetLength(int x)
        {
            return m_matrix.GetLength(x);
        }

        // allow callers to initialize
        public double this[int x, int y]
        {
            get { return m_matrix[x, y]; }
            set { m_matrix[x, y] = value; }
        }
    }
}
