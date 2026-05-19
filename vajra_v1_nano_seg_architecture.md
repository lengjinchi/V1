# Vajra V1 Nano Seg Architecture

对应当前训练脚本里的 `Vajra("vajra-v1-nano-seg")`。

```mermaid
flowchart LR
    subgraph B[Backbone]
        I[Input<br/>3 x 640 x 640]
        B1[Stem1<br/>ConvBNAct 3x3 s2<br/>3 -> 16]
        B2[Stem2<br/>ConvBNAct 3x3 s2<br/>16 -> 32]
        B3[VajraV1MerudandaX<br/>32 -> 64]
        B4[Conv1<br/>ConvBNAct 3x3 s2<br/>64 -> 64]
        B5[VajraV1MerudandaX<br/>64 -> 128]
        B6[Conv2<br/>ConvBNAct 3x3 s2<br/>128 -> 128]
        B7[VajraV1MerudandaX<br/>128 -> 128]
        B8[Conv3<br/>ConvBNAct 3x3 s2<br/>128 -> 256]
        B9[VajraV1MerudandaX<br/>256 -> 256]
        B10[VajraV1AttentionBhag6<br/>SPPF + Attention<br/>256 -> 256]

        I --> B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7 --> B8 --> B9 --> B10
    end

    subgraph N[Neck]
        N1[Upsample x2]
        N2[Concat<br/>B7 + up(B10)]
        N3[VajraV1MerudandaX<br/>384 -> 128]

        N4[Upsample x2]
        N5[Concat<br/>B5 + up(N3)]
        N6[VajraV1MerudandaX<br/>256 -> 64]

        N7[Neck Conv1<br/>ConvBNAct 3x3 s2<br/>64 -> 64]
        N8[Concat<br/>N3 + N7]
        N9[ShapeAdaptiveMerudandaX<br/>192 -> 128<br/>LDConv route]

        N10[Neck Conv2<br/>ConvBNAct 3x3 s2<br/>128 -> 128]
        N11[Concat<br/>B10 + N10]
        N12[VajraV1MerudandaBhag15<br/>384 -> 256]

        B10 --> N1 --> N2 --> N3
        B7 --> N2
        N3 --> N4 --> N5 --> N6
        B5 --> N5
        N6 --> N7 --> N8 --> N9
        N3 --> N8
        N9 --> N10 --> N11 --> N12
        B10 --> N11
    end

    subgraph H[Head]
        P3[P3 / stride 8<br/>N6<br/>64 ch]
        P4[P4 / stride 16<br/>N9<br/>128 ch]
        P5[P5 / stride 32<br/>N12<br/>256 ch]

        F0[Proto branch input<br/>P3]
        F1[FFTGuidedEMA<br/>proto_only = True]
        F2[ProtoMaskModule<br/>64 -> 64 -> upsample -> 32]
        PM[Proto masks]

        S3[Seg coeff branch P3<br/>Conv 64 -> 32 -> 32]
        S4[Seg coeff branch P4<br/>Conv 128 -> 32 -> 32]
        S5[Seg coeff branch P5<br/>Conv 256 -> 32 -> 32]
        MC[Mask coefficients<br/>concat over 3 scales]

        D3[Detect branch P3]
        D4[Detect branch P4]
        D5[Detect branch P5]
        DET[Box + Cls outputs]

        OUT[Segmentation output<br/>DET + MC + PM]

        N6 --> P3
        N9 --> P4
        N12 --> P5

        P3 --> F0 --> F1 --> F2 --> PM

        P3 --> S3 --> MC
        P4 --> S4 --> MC
        P5 --> S5 --> MC

        P3 --> D3 --> DET
        P4 --> D4 --> DET
        P5 --> D5 --> DET

        DET --> OUT
        MC --> OUT
        PM --> OUT
    end
```

## Notes

- 这是代码实际走到的 `nano + seg` 结构，不包含你上一张图里的 `CFSA`，因为当前代码构图时没有把 `CFSA` 接进 `vajra-v1-nano-seg`。
- `ShapeAdaptiveMerudandaX` 出现在 Neck 的第二个下采样融合块里，也就是当前 nano 版最明显的自定义改动。
- 分割 Head 默认是 `use_fft_guided_ema=False`、`use_fft_guided_ema_proto_only=True`，所以现在实际只增强了 `Proto` 分支。
