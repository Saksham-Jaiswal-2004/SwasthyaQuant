# Production Backend Inference Flow

The backend is a direct mirror of the research pipeline and must use the repository's real implementation, not a simplified substitute.

Raw 11 features
    ↓
ClinicalRepresentation
    ↓
8 classical features
    ↓
DCQFExtractor
    ↓
24 quantum features
    ↓
32 hybrid features
    ↓
StandardScaler
    ↓
trained classifier
    ↓
positive-class probability

This exact chain is the only safe production path for the QML model.
