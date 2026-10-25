# H3 public dataset inventory (metadata-only, 2026-09-12)

Target: first-person hand-object interaction video generation.
Required sample: first frame/reference + temporally aligned two-hand control + real target interaction video + caption.

## Remote local scan
- Existing /nvmedata/workspace2/users/nina/dataset_video_train: 3916 metadata rows and 7832 video files reported by a read-only scan.
- Captions are mainly third-person interviews/dance. It is not counted as H3-effective data.
- No Ego4D, EPIC-KITCHENS, or HOI4D directory was found under /nvmedata/workspace2/users/nina at the scan depth.
- No source video was copied by this scan.

## Public source inventory (official totals; effective counts require metadata/license access and filtering)
- Ego4D FHO: first-person hand-object interaction annotations, including pre/contact/PNR/post moments and hand/tool/object boxes. Access requires accepting the dataset license.
- EPIC-KITCHENS-100: first-person kitchen recordings and hand/object annotations; research license is CC BY-NC 4.0.
- HOI4D: 2.4M RGB-D frames over 4000 egocentric sequences, with 3D hand pose and object/action annotations; research license is CC BY-NC 4.0.

## Next read-only step
Obtain/verify official metadata access, then count clips passing: egocentric view, visible hands, interaction contact, object/animal response, continuous frames, and usable license. Only passing manifest rows will be written under manifests/; original videos remain in their source location until explicitly copied.
