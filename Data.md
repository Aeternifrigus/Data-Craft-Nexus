Data is graded and evaluated in a certain manner. Everything in ML Arch and drift monitoring gets decided by Data itself. Data is graded on 6 axes, each axes has a sub grade, and each axes assigns one grade. 

 ==**Hence each data set has 6 grades from 6 axes each and that decides design of Model. Axes 1-3 are fixed and 4-6 can be changed upon how data is arranged.**==

Axes 1=2=3= 0,0,0                                                            Axes 4=5=6= 0,0,0
--- start-multi-column: taxonomy-graphs```column-settings
Number of Columns: 2
Column Size: [50%, 50%]```

```plotly
{
  "data": [
    {"type":"scatter3d","mode":"lines+markers","x":[0,1,2,3,4],"y":[0,0,0,0,0],"z":[0,0,0,0,0],"line":{"color":"#5b5f6b","width":4},"marker":{"color":"#5b5f6b","size":4},"showlegend":false},
    {"type":"scatter3d","mode":"lines+markers","x":[0,0,0,0,0,0],"y":[0,1,2,3,4,5],"z":[0,0,0,0,0,0],"line":{"color":"#5b5f6b","width":4},"marker":{"color":"#5b5f6b","size":4},"showlegend":false},
    {"type":"scatter3d","mode":"lines+markers","x":[0,0,0,0,0,0,0,0],"y":[0,0,0,0,0,0,0,0],"z":[0,1,2,3,4,5,6,7],"line":{"color":"#5b5f6b","width":4},"marker":{"color":"#5b5f6b","size":4},"showlegend":false},
    {"type":"scatter3d","mode":"markers",
     "x":[0,1,2,3,4,1,2,1,1,1,1,2,1,0,0],
     "y":[0,0,0,0,1,1,1,3,3,4,5,0,2,0,0],
     "z":[0,7,3,4,3,0,5,6,6,1,0,7,4,2,4],
     "text":["Housing sale prices","Customer transaction log","Spam-flagged emails","Tagged image subset","LLM next-token training","Stock price history","Speech-to-text transcripts","Social network graph","Molecule structures","Company org chart","LiDAR point cloud","Product listing (text+img+price)","Satellite imagery tiles","Star-rated reviews","Labeled X-ray diagnoses"],
     "marker":{"size":7,"color":["#e0a458","#6fb3b8","#c96a5a","#9d7fc9","#e0a458","#6fb3b8","#c96a5a","#6fb3b8","#7fae6a","#6fb3b8","#6fb3b8","#c96a5a","#6fb3b8","#e0a458","#e0a458"],"opacity":0.95}}
  ],
  "layout": {
    "width": 460, "height": 480,
    "margin":{"l":0,"r":0,"t":0,"b":0},
    "scene": {
      "xaxis":{"title":"Axis 1: Supervision / Label Density","tickmode":"array","tickvals":[0,1,2,3,4],"ticktext":["Labeled","Unlabeled","Partially/weakly labeled","Semi-labeled","Self-labeled"]},
      "yaxis":{"title":"Axis 2: Relational Structure","tickmode":"array","tickvals":[0,1,2,3,4,5],"ticktext":["Independent (i.i.d.)","Sequential / temporal","Spatial","Relational / graph","Hierarchical","Set-based"]},
      "zaxis":{"title":"Axis 3: Modality","tickmode":"array","tickvals":[0,1,2,3,4,5,6,7],"ticktext":["Numeric","Categorical","Ordinal","Text","Image / visual","Audio","Graph-native","Mixed / multimodal"]}
    }
  }
}
```

--- column-end ---

```plotly
{
  "data": [
    {"type":"scatter3d","mode":"lines+markers","x":[0,1,2,3],"y":[0,0,0,0],"z":[0,0,0,0],"line":{"color":"#5b5f6b","width":4},"marker":{"color":"#5b5f6b","size":4},"showlegend":false},
    {"type":"scatter3d","mode":"lines+markers","x":[0,0,0,0,0,0],"y":[0,1,2,3,4,5],"z":[0,0,0,0,0,0],"line":{"color":"#5b5f6b","width":4},"marker":{"color":"#5b5f6b","size":4},"showlegend":false},
    {"type":"scatter3d","mode":"lines+markers","x":[0,0,0,0],"y":[0,0,0,0],"z":[0,1,2,3],"line":{"color":"#5b5f6b","width":4},"marker":{"color":"#5b5f6b","size":4},"showlegend":false},
    {"type":"scatter3d","mode":"markers",
     "x":[0,1,2,2,3,0,0,3,0,0,0,2,0],
     "y":[0,2,0,1,0,1,3,2,5,0,4,0,3],
     "z":[0,0,0,2,0,0,3,1,1,0,0,2,0],
     "text":["Clean tabular dataset (5 feat, 10k rows)","Gene expression microarray","Bag-of-words text vectors","User-item rating matrix","Pixel image data","Fraud detection transactions","Financial market data","Sensor readings with dropout","Survey with income question skipped","Manually entered survey (typos)","Linearly separable structured data (e.g. Iris)","One-hot encoded categories","Consumer behavior stream (drift)"],
     "marker":{"size":7,"color":["#e0a458","#6fb3b8","#c96a5a","#9d7fc9","#7fae6a","#6fb3b8","#c96a5a","#6fb3b8","#9d7fc9","#c96a5a","#e0a458","#c96a5a","#6fb3b8"],"opacity":0.95}}
  ],
  "layout": {
    "width": 460, "height": 480,
    "margin":{"l":0,"r":0,"t":0,"b":0},
    "scene": {
      "xaxis":{"title":"Axis 4: Dimensionality & Scale","tickmode":"array","tickvals":[0,1,2,3],"ticktext":["Low-dimensional","High-dimensional","Sparse","Dense"]},
      "yaxis":{"title":"Axis 5: Distributional Properties","tickmode":"array","tickvals":[0,1,2,3,4,5],"ticktext":["Balanced","Imbalanced","Stationary","Non-stationary","Linearly separable","Non-linearly separable"]},
      "zaxis":{"title":"Axis 6: Missingness / Quality","tickmode":"array","tickvals":[0,1,2,3],"ticktext":["Complete","Missing at random","Missing not at random","Noisy"]}
    }
  }
}
```

--- end-multi-column




## Axis 1: Supervision / Label Density [A1]

| Type                              | Description                                               | Example                                                                   |
| --------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------- |
| Labeled <br>[A11]                 | Every example has a known target                          | Housing prices with sale amount attached                                  |
| Unlabeled <br>[A12]               | No targets at all                                         | Raw customer transaction logs                                             |
| Partially/weakly labeled<br>[A13] | Labels exist but are coarse, noisy, or indirect           | Email marked "spam" but no word-level annotation                          |
| Semi-labeled [A14]                | Small labeled subset + large unlabeled pool               | 500 tagged images + 50,000 untagged                                       |
| Self-labeled <br>[A15]            | Labels manufactured from the data itself (a pretext task) | Predicting the next word : the "label" is the next token, already present |

## Axis 2: Relational Structure [A2]

| Type                            | Description                                            | Example                                           |
| ------------------------------- | ------------------------------------------------------ | ------------------------------------------------- |
| Independent (i.i.d.) [A21]      | Rows are unrelated; shuffling changes nothing          | Customer records, one row per transaction         |
| Sequential / temporal <br>[A22] | Order matters; position depends on what came before    | Stock prices over time, sentences, sensor streams |
| Spatial<br>[A23]                | Neighboring values in 2D/3D space are related          | Images, audio spectrograms, satellite maps        |
| Relational / graph <br>[A24]    | Points explicitly linked to specific other points      | Social networks, molecules, supply chains         |
| Hierarchical <br>[A25]          | Nested containment : points belong inside other points | Org charts, file systems, taxonomies              |
| Set-based<br>[A26]              | Elements interact but have no inherent order           | Point clouds, a "bag" of items in a cart          |
## Axis 3: Modality [A3]

| Type                           | Description                                        | Example                                     |
| ------------------------------ | -------------------------------------------------- | ------------------------------------------- |
| Numeric : continuous<br>[A31]  | Real-valued measurements                           | Price, temperature, weight                  |
| Categorical : nominal<br>[A32] | Unordered discrete labels                          | Country, product category                   |
| Ordinal<br>[A33]               | Discrete but ordered                               | Star rating, education level                |
| Text<br>[A34]                  | Sequences of discrete tokens with semantic meaning | Reviews, emails, transcripts                |
| Image / visual<br>[A35]        | Pixel grids                                        | Photos, X-rays                              |
| Audio<br>[A36]                 | Waveform / spectral signal                         | Speech, music                               |
| Graph-native<br>[A37]          | Nodes + edges as the raw data itself               | Molecule structures, network topology       |
| Mixed / multimodal<br>[A38]    | Combination of the above in one dataset            | A product listing with text + image + price |
## Axis 4: Dimensionality & Scale [A4]

| Type                      | Description                                | Example                                       |
| ------------------------- | ------------------------------------------ | --------------------------------------------- |
| Low-dimensional<br>[A41]  | Few features relative to number of samples | 5 features, 10,000 rows                       |
| High-dimensional<br>[A42] | Many features, possibly more than samples  | Gene expression data, bag-of-words text       |
| Sparse<br>[A43]           | Most values are zero/missing               | One-hot encoded categories, user-item ratings |
| Dense<br>[A44]            | Most values are populated and meaningful   | Pixel data, sensor readings                   |

## Axis 5: Distributional Properties [A5]

| Type                            | Description                                   | Example                                          |
| ------------------------------- | --------------------------------------------- | ------------------------------------------------ |
| Balanced<br>[A51]               | Classes/outcomes occur roughly equally        | Most default assumptions hold                    |
| Imbalanced<br>[A52]             | One outcome dominates                         | Fraud detection (99%+ non-fraud)                 |
| Stationary<br>[A53]             | Statistical properties don't change over time | Physical measurements under stable conditions    |
| Non-stationary<br>[A54]         | Distribution shifts over time                 | Consumer behavior, financial markets (= "drift") |
| Linearly separable<br>[A55]     | Classes can be split by a straight line/plane | Simple structured data                           |
| Non-linearly separable<br>[A56] | Requires curved boundaries to separate        | Most real-world data                             |
## Axis 6: Missingness / Quality [A6]

| Type                           | Description                               | Example                                             |
| ------------------------------ | ----------------------------------------- | --------------------------------------------------- |
| Complete<br>[A61]              | No missing values                         | Clean sensor log                                    |
| Missing at random<br>[A62]     | Gaps unrelated to the value itself        | Random sensor dropout                               |
| Missing not at random<br>[A63] | Gaps that are informative                 | People skip income question because of their income |
| Noisy<br>[A64]                 | Values present but corrupted or imprecise | Manually entered survey data with typos             |
