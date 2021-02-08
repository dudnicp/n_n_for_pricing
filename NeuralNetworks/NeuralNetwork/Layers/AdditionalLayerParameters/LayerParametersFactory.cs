using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers.AdditionalLayerParameters
{
    public static class LayerParametersFactory
    {
        public static IAdditionalLayerParameters Build(LayerType type)
        {
            switch (type)
            {
                case LayerType.Standard:
                    return new StandardParameters();
                case LayerType.Dropout:
                    return new DropoutParameters();
                case LayerType.L2Penalty:
                    return new L2PenaltyParameters();
                case LayerType.WeightDecay:
                    return new WeightDecayParameters();
                case LayerType.InputStandardizing:
                    return new InputStandardizingParameters();
                default:
                    throw new InvalidOperationException("Unknown parameters type: " + type.ToString());
            }
        }
    }
}
