import React, { useState, useCallback } from 'react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [taxonomy, setTaxonomy] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [showSessions, setShowSessions] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('all');

  // Handle file drag and drop
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const fileList = Array.from(e.dataTransfer.files).filter(file => 
        file.type.startsWith('image/')
      );
      setFiles(fileList);
    }
  }, []);

  const handleFileSelect = (e) => {
    if (e.target.files) {
      const fileList = Array.from(e.target.files).filter(file => 
        file.type.startsWith('image/')
      );
      setFiles(fileList);
    }
  };

  const uploadImages = async () => {
    if (files.length === 0) return;

    setUploading(true);
    setTaxonomy(null);

    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });

      const response = await fetch(`${API_BASE_URL}/api/upload-images`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload images');
      }

      const result = await response.json();
      setTaxonomy(result.taxonomy);
      setSessionId(result.session_id);
    } catch (error) {
      console.error('Error uploading images:', error);
      alert('Error uploading images. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const loadSessions = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/sessions`);
      const data = await response.json();
      setSessions(data.sessions);
      setShowSessions(true);
    } catch (error) {
      console.error('Error loading sessions:', error);
    }
  };

  const loadSession = async (sessionId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/taxonomy/${sessionId}`);
      const data = await response.json();
      setTaxonomy(data.taxonomy);
      setSessionId(sessionId);
      setShowSessions(false);
    } catch (error) {
      console.error('Error loading session:', error);
    }
  };

  const clearFiles = () => {
    setFiles([]);
    setTaxonomy(null);
    setSessionId(null);
  };

  const getFilteredHashtags = () => {
    if (!taxonomy?.deduplicated_hashtags) return {};
    
    if (selectedCategory === 'all') {
      return taxonomy.deduplicated_hashtags;
    }
    
    return {
      [selectedCategory]: taxonomy.deduplicated_hashtags[selectedCategory] || []
    };
  };

  const getTotalHashtags = () => {
    if (!taxonomy?.deduplicated_hashtags) return 0;
    return Object.values(taxonomy.deduplicated_hashtags).reduce((sum, arr) => sum + arr.length, 0);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                🎨 AI Image Taxonomy Generator
              </h1>
              <p className="text-gray-600 mt-1">
                Upload images and let AI create intelligent hashtag taxonomies
              </p>
            </div>
            <button
              onClick={loadSessions}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
            >
              View Sessions
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Sessions Modal */}
        {showSessions && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-xl p-6 max-w-2xl w-full mx-4 max-h-96 overflow-y-auto">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold">Previous Sessions</h3>
                <button
                  onClick={() => setShowSessions(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>
              <div className="space-y-3">
                {sessions.map((session) => (
                  <div
                    key={session.session_id}
                    className="p-3 border rounded-lg hover:bg-gray-50 cursor-pointer"
                    onClick={() => loadSession(session.session_id)}
                  >
                    <div className="flex justify-between items-center">
                      <div>
                        <p className="font-medium">
                          {session.image_count} images processed
                        </p>
                        <p className="text-sm text-gray-500">
                          {new Date(session.created_at).toLocaleDateString()}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-sm text-purple-600">
                          {session.hashtag_count?.deduplicated_total || 0} hashtags
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Upload Section */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold mb-6 text-gray-800">Upload Images</h2>
          
          {/* File Drop Zone */}
          <div
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
              dragActive 
                ? 'border-purple-500 bg-purple-50' 
                : 'border-gray-300 hover:border-purple-400'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <div className="text-6xl mb-4">📁</div>
            <h3 className="text-xl font-semibold mb-2">
              Drag & drop images here
            </h3>
            <p className="text-gray-600 mb-4">
              or click to select files
            </p>
            <input
              type="file"
              multiple
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="inline-block px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 cursor-pointer transition-colors"
            >
              Choose Files
            </label>
          </div>

          {/* Selected Files */}
          {files.length > 0 && (
            <div className="mt-6">
              <h4 className="font-semibold mb-3">Selected Files ({files.length})</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                {files.slice(0, 8).map((file, index) => (
                  <div key={index} className="bg-gray-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-600 truncate">
                      {file.name}
                    </div>
                    <div className="text-xs text-gray-500">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </div>
                  </div>
                ))}
                {files.length > 8 && (
                  <div className="bg-gray-100 p-3 rounded-lg flex items-center justify-center">
                    <span className="text-gray-600">+{files.length - 8} more</span>
                  </div>
                )}
              </div>
              
              <div className="flex gap-3">
                <button
                  onClick={uploadImages}
                  disabled={uploading}
                  className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                    uploading
                      ? 'bg-gray-400 text-white cursor-not-allowed'
                      : 'bg-green-600 text-white hover:bg-green-700'
                  }`}
                >
                  {uploading ? (
                    <span className="flex items-center">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Analyzing Images...
                    </span>
                  ) : (
                    'Analyze Images'
                  )}
                </button>
                
                <button
                  onClick={clearFiles}
                  className="px-6 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
                >
                  Clear
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Results Section */}
        {taxonomy && (
          <div className="bg-white rounded-xl shadow-lg p-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-800">
                Generated Taxonomy
              </h2>
              <div className="text-sm text-gray-600">
                Session: {sessionId?.slice(0, 8)}...
              </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-purple-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {taxonomy.hashtag_count?.deduplicated_total || 0}
                </div>
                <div className="text-sm text-gray-600">Final Hashtags</div>
              </div>
              <div className="bg-blue-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">
                  {taxonomy.hashtag_count?.original_total || 0}
                </div>
                <div className="text-sm text-gray-600">Original Features</div>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-green-600">
                  {taxonomy.hashtag_count?.duplicates_removed || 0}
                </div>
                <div className="text-sm text-gray-600">Duplicates Removed</div>
              </div>
            </div>

            {/* Category Filter */}
            <div className="mb-6">
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => setSelectedCategory('all')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    selectedCategory === 'all'
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  All Categories
                </button>
                {Object.keys(taxonomy.deduplicated_hashtags || {}).map((category) => (
                  <button
                    key={category}
                    onClick={() => setSelectedCategory(category)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors capitalize ${
                      selectedCategory === category
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {category} ({taxonomy.deduplicated_hashtags[category]?.length || 0})
                  </button>
                ))}
              </div>
            </div>

            {/* Hashtags Display */}
            <div className="space-y-6">
              {Object.entries(getFilteredHashtags()).map(([category, hashtags]) => (
                <div key={category}>
                  <h3 className="text-lg font-semibold capitalize mb-3 text-gray-800">
                    {category} ({hashtags.length})
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {hashtags.map((hashtag, index) => (
                      <span
                        key={index}
                        className="px-3 py-1 bg-gradient-to-r from-purple-100 to-blue-100 text-gray-800 rounded-full text-sm font-medium border border-purple-200"
                      >
                        #{hashtag}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Semantic Groups */}
            {taxonomy.semantic_groups && taxonomy.semantic_groups.length > 0 && (
              <div className="mt-8 pt-6 border-t">
                <h3 className="text-lg font-semibold mb-4">Semantic Groups</h3>
                <div className="space-y-4">
                  {taxonomy.semantic_groups.map((group, index) => (
                    <div key={index} className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="font-medium mb-2">{group.group_name}</h4>
                      <p className="text-sm text-gray-600 mb-3">{group.description}</p>
                      <div className="flex flex-wrap gap-2">
                        {group.hashtags.map((hashtag, idx) => (
                          <span
                            key={idx}
                            className="px-2 py-1 bg-white text-gray-700 rounded text-sm border"
                          >
                            #{hashtag}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
