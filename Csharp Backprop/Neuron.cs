using System;
using System.Collections.Generic;

namespace Csharp_Backprop
{
    internal class Neuron
    {
        // Neuron variables. these should be set to private and only accessed using a property.
        private List<double> weights = new List<double> { };

        private List<double> inputs = new List<double> { };
        private List<double> weight_Update = new List<double> { };
        private List<double> weight_Update_Old = new List<double> { };
        private double output = 0;
        private double error = 0;

        // Constructor
        public Neuron(int Number_Of_Connections, int Seed)
        {
            Random RN = new Random(Seed);

            for (int i = 0; i < Number_Of_Connections; i++)
            {
                this.weights.Add(RN.NextDouble() * 2.0 - 1.0);
                this.inputs.Add(0);
                this.weight_Update.Add(0);
                this.weight_Update_Old.Add(0);
            }

            // Add additional value to represent bias
            this.weights.Add(RN.NextDouble());
            this.inputs.Add(0);
            this.weight_Update.Add(0);
            this.weight_Update_Old.Add(0);
        }

        // properties
        public List<double> Weights
        {
            get { return weights; }
            set { weights = value; }
        }

        public double Output
        {
            get { return output; }
            set { output = value; }
        }

        public List<double> Inputs
        {
            get { return inputs; }
            set { inputs = value; }
        }

        public double Error
        {
            get { return error; }
            set { error = value; }
        }

        public List<double> Weight_Update
        {
            get { return weight_Update; }
            set { weight_Update = value; }
        }

        public List<double> Weight_Update_Old
        {
            get { return weight_Update_Old; }
            set { weight_Update_Old = value; }
        }
    }
}