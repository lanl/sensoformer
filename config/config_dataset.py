import os

class DataConfig(dict):
    def __init__(self, project_root: str):
        super().__init__({
            # -------------- Pretraining datasets ------------------
            'LANSCE_pt': {
                'root': os.path.join(project_root, 'datasets', 'LANSCE', 'processed_h5', 'train'),
                'n_files': 6
            },
            'LHC_pt': {
                'root': os.path.join(project_root, 'datasets', 'LHC', 'processed_h5', 'train'),
                'n_files': 2
            },
            'RCS_Fermilab_pt': {
                'root': os.path.join(project_root, 'datasets', 'RCS_Fermilab', 'processed_h5', 'train'),
                'n_files': 0
            },
            'HVCM_SNS_pt': {
                'root': os.path.join(project_root, 'datasets', 'HVCM_SNS', 'processed_h5', 'train'),
                'n_files': 2
            },
            'LCLS_SLAC_pt': {
                'root': os.path.join(project_root, 'datasets', 'LCLS_SLAC', 'processed_h5', 'train'),
                'n_files': 3
            },
            'SRF_CEBAF_pt': {
                'root': os.path.join(project_root, 'datasets', 'SRF_CEBAF', 'processed_h5', 'train'),
                'n_files': 1
            },
            'TCV_tokamak_pt': {
                'root': os.path.join(project_root, 'datasets', 'TCV_tokamak', 'processed_h5', 'train'),
                'n_files': 0
            },

            # -------------- Downstream datasets ------------------
            'LANSCE_dt_1': {
                'root': os.path.join(project_root, 'datasets', 'LANSCE', 'processed_h5', 'test', '2019-2024'),
                'n_files': 6
            },
            'LANSCE_dt_2': {
                'root': os.path.join(project_root, 'datasets', 'LANSCE', 'processed_h5', 'test', '2025'),
                'n_files': 1
            },
            'LHC_dt': {
                'root': os.path.join(project_root, 'datasets', 'LHC', 'processed_h5', 'test'),
                'n_files': 2
            },
            'HVCM_SNS_dt': {
                'root': os.path.join(project_root, 'datasets', 'HVCM_SNS', 'processed_h5', 'test'),
                'n_files': 21
            },
            'LCLS_SLAC_dt': {
                'root': os.path.join(project_root, 'datasets', 'LCLS_SLAC', 'processed_h5', 'test'),
                'n_files': 1
            },
            'SRF_CEBAF_dt': {
                'root': os.path.join(project_root, 'datasets', 'SRF_CEBAF', 'processed_h5', 'test'),
                'n_files': 1
            },
            # ----------- New unseen accelerator -----------------
            'RCS_Fermilab_dt': {
                'root': os.path.join(project_root, 'datasets', 'RCS_Fermilab', 'processed_h5', 'test'),
                'n_files': 1
            },
            # ----------- New unseen scientific facility -----------------
            'TCV_tokamak_dt': {
                'root': os.path.join(project_root, 'datasets', 'TCV_tokamak', 'processed_h5', 'test'),
                'n_files': 1
            },
        })

    def collect_h5_paths(self, dataset_keys):
        h5_paths = []
        per_dataset_counts = {}

        for key in dataset_keys:  # e.g. "LANSCE_pt", "LHC_pt", etc.
            if key not in self:
                raise KeyError(f"Dataset key not found in DataConfig: {key}")

            root = self[key]["root"]
            n_files = int(self[key].get("n_files", 0))

            if not os.path.isdir(root):
                raise FileNotFoundError(f"Directory does not exist: {root}")

            # files inside every dataset directory should be .hdf5 or .h5
            files = sorted(os.path.join(root, f) for f in os.listdir(root)
                                if f.endswith(".hdf5") or f.endswith(".h5")) 

            # n_files = 0 means use all files
            if n_files > 0:
                files = files[:n_files]

            if len(files) == 0:
                raise FileNotFoundError(f"No .hdf5 or .h5 files found in {root} for key {key}")

            h5_paths.extend(files)
            per_dataset_counts[key] = len(files)

        return h5_paths, per_dataset_counts