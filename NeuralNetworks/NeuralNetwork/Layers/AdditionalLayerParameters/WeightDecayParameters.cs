using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers.AdditionalLayerParameters
{
    class WeightDecayParameters : IAdditionalLayerParameters
    {
        public LayerType LayerType => LayerType.WeightDecay;

        public double DecayRate { get; set; }

        public WeightDecayParameters()
        {
        }
    }
}
