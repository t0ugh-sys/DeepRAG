import json

from backend.document_manager import DocumentMetadata


def test_document_metadata_save_load_atomic(tmp_path):
    metadata_file = tmp_path / 'documents_metadata.json'

    metadata = DocumentMetadata(metadata_file=str(metadata_file))
    metadata.update_document_info('a.txt', {'tags': ['x'], 'category': 'test'})

    assert metadata_file.exists()
    parsed = json.loads(metadata_file.read_text(encoding='utf-8'))
    assert 'a.txt' in parsed

    # The atomic save path should not leak temp files.
    assert list(tmp_path.glob(metadata_file.name + '.*.tmp')) == []

    reloaded = DocumentMetadata(metadata_file=str(metadata_file))
    assert reloaded.get_document_info('a.txt') is not None

