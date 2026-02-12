#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>
#include <set>
#include <vector>

using namespace std;

// to store type and value
struct Token
{
    string type;
    string value;
};

set<string> keywords = {"let", "where", "true", "false", "not", "fn", "ls", "gr", "ge", "aug", "le", "nil", "dummy", "or", "in", "eq", "ne", "and", "rec", "within"};

bool isLetter(char ch)
{
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
}

bool isDigit(char ch)
{
    return ch >= '0' && ch <= '9';
}

bool isOperatorSymbol(char ch)
{
    string ops = "+-*<>&.@/:=~|$!#%^_[]{}\"'?";
    return ops.find(ch) != string::npos;
}

vector<Token> tokenize(ifstream &stream)
{
    vector<Token> tokens; // type, value
    char ch; // to char by char check

    while (stream.get(ch))
    {
        if (ch == ' ' || ch == '\t' || ch == '\n')
            continue;
        
        // -- Case 01 : Check "KEYWORD" or "IDENTIFIER" --
        if (isLetter(ch) || isDigit(ch))
        {
            string value;
            value += ch;
            while (stream.get(ch) && (isLetter(ch) || isDigit(ch) || ch == '_'))
            {
                value += ch;
            }
            stream.unget();
            string type = keywords.find(value) != keywords.end() ? "KEYWORD" : "IDENTIFIER";
            tokens.push_back({type, value});
        }

        // -- Case 02 : Check "INTEGER" --
        else if (isDigit(ch))
        {
            string value;
            value += ch;
            while (stream.get(ch) && isDigit(ch))
            {
                value += ch;
            }
            stream.unget();
            tokens.push_back({"INTEGER", value});
        }

        // -- Case 03 : Check "STRING" --
        else if (ch == '\'')
        {
            string value;
            value += ch;
            while (stream.get(ch) && ch != '\'')
            {
                value += ch;
            }
            value += ch;
            tokens.push_back({"STRING", value});
        }

        // -- Case 04 : Check "PUNCTUATION" --
        else if (ch == '(' || ch == ')' || ch == ';' || ch == ',')
        {
            tokens.push_back({"PUNCTUATION", string(1, ch)});
        }

        // -- Case 05 : Check "OPERATOR" --
        else if (isOperatorSymbol(ch))
        {
            string value;
            value += ch;
            tokens.push_back({"OPERATOR", value});
        }
    }

    return tokens;
}

int main(int argc, char *argv[])
{
    // --- Check the input correctness ---
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <filename>" << endl;
        return EXIT_FAILURE;
    }
    
    // --- file read and start the program ---
    ifstream stream(argv[1]);
    
    if (!stream.is_open())
    {
        cerr << "Error in opening the file!" << endl;
        return EXIT_FAILURE;
    }

    vector<Token> tokens = tokenize(stream);
    for (const auto &token : tokens)
    {
        cout << "<" << token.type << ", \"" << token.value << "\">" << endl;
    }

    stream.close();
    return EXIT_SUCCESS;
}