using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers.AdditionalLayerParameters
{
    public class L2PenaltyParameters : IAdditionalLayerParameters
    {
        public LayerType LayerType => LayerType.L2Penalty;

        public double PenaltyCoefficient { get; set; }

        public L2PenaltyParameters()
        {
        }
    }
}
