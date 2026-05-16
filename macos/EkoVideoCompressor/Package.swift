// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "EkoVideoCompressor",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "EkoVideoCompressor", targets: ["EkoVideoCompressor"])
    ],
    targets: [
        .executableTarget(
            name: "EkoVideoCompressor",
            path: "EkoVideoCompressor"
        )
    ]
)
