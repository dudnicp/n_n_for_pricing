using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers.AdditionalLayerParameters
{
    public class DropoutParameters : IAdditionalLayerParameters
    {
        public LayerType LayerType => LayerType.Dropout;

        public double KeepProbability { get; set; }

        public DropoutParameters()
        {
        }
    }
}
